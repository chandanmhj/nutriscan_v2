"""
NutriScan v2.0 — Fully on Groq (free, no Gemini)
- Vision: llama-4-scout for image analysis
- Chat: llama-3.3-70b for casual conversation
- 1 API call per photo
"""

import os, sqlite3, json, logging, asyncio, base64, re
from datetime import date
from io import BytesIO

from groq import AsyncGroq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters, ConversationHandler
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("nutriscan")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "YOUR_GROQ_API_KEY")
DB_PATH        = os.getenv("DB_PATH", "nutriscan.db")
DEFAULT_GOAL   = 2000

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]
CHAT_MODEL = "llama-3.3-70b-versatile"

# ── Conversation states ────────────────────────────────────────────────────────
ASK_NAME, ASK_AGE, ASK_HEIGHT, ASK_WEIGHT, ASK_DIABETES, ASK_DIABETES_LEVEL = range(6)
WAITING_QUANTITY = 10

# ── DB ─────────────────────────────────────────────────────────────────────────
def get_db():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

def init_db():
    with get_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
            height_cm REAL, weight_kg REAL, bmi REAL, is_obese INTEGER DEFAULT 0,
            diabetes TEXT DEFAULT 'none', diabetes_level TEXT,
            daily_goal INTEGER DEFAULT 2000, onboarded INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS food_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
            log_date TEXT NOT NULL, food_name TEXT, calories REAL,
            protein_g REAL, carbs_g REAL, fat_g REAL, fiber_g REAL, sugar_g REAL,
            quantity TEXT, image_type TEXT, advice TEXT,
            logged_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (chat_id) REFERENCES users(chat_id)
        );
        """)

def ensure_user(chat_id):
    with get_db() as c: c.execute("INSERT OR IGNORE INTO users (chat_id) VALUES (?)", (chat_id,))

def get_user(chat_id):
    with get_db() as c:
        r = c.execute("SELECT * FROM users WHERE chat_id=?", (chat_id,)).fetchone()
    return dict(r) if r else None

def update_user(chat_id, **kw):
    if not kw: return
    cols = ", ".join(f"{k}=?" for k in kw)
    with get_db() as c: c.execute(f"UPDATE users SET {cols} WHERE chat_id=?", list(kw.values())+[chat_id])

def add_entry(chat_id, data, image_type, qty, advice=""):
    ensure_user(chat_id)
    with get_db() as c:
        c.execute("""INSERT INTO food_log
            (chat_id,log_date,food_name,calories,protein_g,carbs_g,fat_g,fiber_g,sugar_g,quantity,image_type,advice)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (chat_id, date.today().isoformat(), data.get("food_name","Unknown"),
             data.get("calories",0), data.get("protein_g",0), data.get("carbs_g",0),
             data.get("fat_g",0), data.get("fiber_g"), data.get("sugar_g"),
             qty, image_type, advice))

def get_today_log(chat_id):
    with get_db() as c:
        rows = c.execute("SELECT * FROM food_log WHERE chat_id=? AND log_date=? ORDER BY logged_at",
            (chat_id, date.today().isoformat())).fetchall()
    return [dict(r) for r in rows]

def get_today_totals(chat_id):
    with get_db() as c:
        r = c.execute("""SELECT COALESCE(SUM(calories),0) AS calories,
            COALESCE(SUM(protein_g),0) AS protein, COALESCE(SUM(carbs_g),0) AS carbs,
            COALESCE(SUM(fat_g),0) AS fat, COALESCE(SUM(sugar_g),0) AS sugar,
            COUNT(*) AS entries FROM food_log WHERE chat_id=? AND log_date=?""",
            (chat_id, date.today().isoformat())).fetchone()
    return dict(r) if r else {}

def clear_today(chat_id):
    with get_db() as c: c.execute("DELETE FROM food_log WHERE chat_id=? AND log_date=?", (chat_id, date.today().isoformat()))

def delete_last(chat_id):
    with get_db() as c:
        r = c.execute("SELECT id FROM food_log WHERE chat_id=? AND log_date=? ORDER BY logged_at DESC LIMIT 1",
            (chat_id, date.today().isoformat())).fetchone()
        if r: c.execute("DELETE FROM food_log WHERE id=?", (r["id"],)); return True
    return False

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_bmi(h_cm, w_kg):
    h = h_cm / 100; bmi = round(w_kg / (h*h), 1); return bmi, bmi >= 30.0

def health_profile(u):
    diab = u.get("diabetes","none"); dl = u.get("diabetes_level")
    d_str = "None" if diab=="none" else diab.title()+(f"/{dl}" if dl else "")
    return (f"Name:{u.get('name')} Age:{u.get('age')} H:{u.get('height_cm')}cm "
            f"W:{u.get('weight_kg')}kg BMI:{u.get('bmi')}({'obese' if u.get('is_obese') else 'normal'}) "
            f"Diabetes:{d_str} DailyGoal:{u.get('daily_goal')}kcal")

def today_log_str(chat_id):
    entries = get_today_log(chat_id)
    if not entries: return "No food logged today."
    t = get_today_totals(chat_id)
    lines = [f"- {e['food_name']}: {e['calories']:.0f}kcal ({e['quantity']})" for e in entries]
    lines.append(f"Total: {t.get('calories',0):.0f}kcal sugar:{t.get('sugar',0):.0f}g")
    return "\n".join(lines)

def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```[a-z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return json.loads(t.strip())

def pbar(cur, goal, w=10):
    f = min(int((cur/goal)*w), w) if goal else 0
    return f"[{'█'*f}{'░'*(w-f)}] {int((cur/goal)*100) if goal else 0}%"

def fmt_summary(chat_id, u):
    t = get_today_totals(chat_id); goal = u.get("daily_goal", DEFAULT_GOAL); cal = t.get("calories",0)
    over = max(cal-goal,0); rem = max(goal-cal,0)
    return "\n".join([
        f"📊 *{u.get('name')}'s Summary* — {date.today().strftime('%d %b %Y')}","",
        f"🔥 *Calories:* `{cal:.0f}` / `{goal}` kcal", f"    {pbar(cal,goal)}",
        f"    {'⚠️ Over by `'+str(int(over))+'` kcal' if over else '✅ `'+str(int(rem))+'` kcal left'}","",
        f"🥩`{t.get('protein',0):.1f}g` 🍞`{t.get('carbs',0):.1f}g` 🧈`{t.get('fat',0):.1f}g` 🍬`{t.get('sugar',0):.1f}g`",
        f"📝 Entries: `{t.get('entries',0)}`"])

VERDICT_EMOJI = {
    "excellent": "🌟", "good": "✅", "moderate": "🟡", "poor": "🔴", "avoid": "☠️"
}

def fmt_result(data, qty=None):
    """Rich formatter matching the old NutriScan style — works for both LABEL and FOOD."""
    itype      = data.get("type", "FOOD")
    name       = data.get("food_name", "This product")
    verdict    = data.get("overall_verdict", "moderate")
    v_emoji    = VERDICT_EMOJI.get(verdict, "🟡")
    short_sum  = data.get("short_summary", "")
    long_adv   = data.get("long_term_advice", "")
    serving    = data.get("serving_size", "")
    svgs       = data.get("servings_per_container")

    cal   = data.get("calories", 0)
    fat   = data.get("total_fat_g")
    s_fat = data.get("saturated_fat_g")
    t_fat = data.get("trans_fat_g")
    chol  = data.get("cholesterol_mg")
    sod   = data.get("sodium_mg")
    carbs = data.get("total_carbs_g") or data.get("carbs_g")
    fiber = data.get("dietary_fiber_g") or data.get("fiber_g")
    sugar = data.get("sugar_g")
    prot  = data.get("protein_g", 0)

    SEP = "──────────────────────"

    lines = []

    # Header
    if itype == "LABEL":
        svgs_txt = f"{svgs} servings × {serving}" if svgs and serving else (serving or "")
        lines += [f"🏷️ *{name}*", f"_{svgs_txt}_" if svgs_txt else "", ""]
    else:
        conf_e = {"low":"🔴","medium":"🟡","high":"🟢"}.get(data.get("confidence","medium"),"🟡")
        lines += [f"🍽️ *{name}*", f"_Quantity: {qty}_ {conf_e}", ""]

    # Verdict + summary
    lines += [
        f"{v_emoji}  *Overall rating: {verdict.title()}*",
        "",
        short_sum if short_sum else "",
        "",
    ]

    # Nutrition table
    per_label = "per serving" if itype == "LABEL" else f"for {qty}"
    lines += [f"*Nutrition Facts ({per_label})*", SEP]
    lines.append(f"Calories: *{cal:.0f}*")
    if fat       is not None: lines.append(f"Total fat: {fat:.1f}g")
    if s_fat     is not None: lines.append(f"  Saturated fat: {s_fat:.1f}g")
    if t_fat     is not None: lines.append(f"  Trans fat: {t_fat:.1f}g")
    if chol      is not None: lines.append(f"Cholesterol: {chol:.0f}mg")
    if sod       is not None: lines.append(f"Sodium: {sod:.0f}mg")
    if carbs     is not None: lines.append(f"Total carbs: {carbs:.1f}g")
    if fiber     is not None: lines.append(f"  Dietary fiber: {fiber:.1f}g")
    if sugar     is not None: lines.append(f"  Sugar: {sugar:.1f}g")
    lines.append(f"Protein: {prot:.1f}g")
    lines.append(SEP)

    # Health alerts
    alerts = []
    if sugar  is not None and sugar  > 15: alerts.append(f"🍬 Sugar is very high ({sugar:.1f}g) — limit to avoid health risks.")
    if sod    is not None and sod    > 600: alerts.append(f"🧂 Sodium is very high ({sod:.0f}mg) — watch your intake.")
    if fat    is not None and fat    > 20: alerts.append(f"🧈 Fat is high ({fat:.1f}g) — consume sparingly.")
    if t_fat  is not None and t_fat  > 0:  alerts.append(f"⚠️ Contains trans fat ({t_fat:.1f}g) — avoid if possible.")
    if verdict == "avoid":                  alerts.append("🚫 This product is not recommended for regular consumption.")
    if alerts:
        lines += ["", "*Health Alerts*"] + alerts

    # Long-term advice
    if long_adv:
        lines += ["", f"*Long-term:* _{long_adv}_"]

    lines += ["", "_Values are per serving. Consult a doctor for personalised advice._"]
    return "\n".join(l for l in lines if l is not None)

# ── AI calls (all Groq, fully free) ───────────────────────────────────────────
def build_photo_prompt(user, log_str):
    return f"""You are NutriScan, a nutrition assistant. Analyze this image carefully.

Return ONLY a valid JSON object, no markdown, no explanation.

If this is a NUTRITION LABEL, return:
{{"type":"LABEL","food_name":"product name","serving_size":"e.g. 200ml","servings_per_container":null,"calories":0,"total_fat_g":0,"saturated_fat_g":null,"trans_fat_g":null,"cholesterol_mg":null,"sodium_mg":null,"total_carbs_g":0,"dietary_fiber_g":null,"sugar_g":null,"protein_g":0,"overall_verdict":"excellent|good|moderate|poor|avoid","short_summary":"1-2 sentence plain English diet advice personalized to user","long_term_advice":"1 sentence about daily consumption impact","advice":"1-2 sentence fun clingy personalized reaction using their name"}}

If this is a FOOD or MEAL photo, return:
{{"type":"FOOD","food_name":"name of food","calories":0,"total_fat_g":0,"saturated_fat_g":null,"sodium_mg":null,"total_carbs_g":0,"dietary_fiber_g":null,"sugar_g":null,"protein_g":0,"confidence":"low|medium|high","overall_verdict":"excellent|good|moderate|poor|avoid","short_summary":"1-2 sentence diet advice personalized to user","long_term_advice":"1 sentence about daily consumption impact","advice":"ask how much they ate in a fun clingy 1 sentence"}}

Verdict guide: excellent=very healthy daily food, good=mostly healthy, moderate=okay occasionally, poor=limit consumption, avoid=very high sugar/sodium/fat.

User profile: {health_profile(user)}
Today so far: {log_str}

Advice/summary rules: use their first name, be dramatic if diabetic and food is high sugar, warn if going over daily calorie goal, be encouraging for healthy choices."""

def build_food_qty_prompt(user, log_str, qty):
    return f"""You are NutriScan. This is a food photo. The person ate: "{qty}".

Estimate nutrition for that quantity and return ONLY valid JSON, no markdown:
{{"type":"FOOD","food_name":"...","calories":0,"total_fat_g":0,"saturated_fat_g":null,"sodium_mg":null,"total_carbs_g":0,"dietary_fiber_g":null,"sugar_g":null,"protein_g":0,"confidence":"low|medium|high","overall_verdict":"excellent|good|moderate|poor|avoid","short_summary":"1-2 sentence diet advice personalized to user","long_term_advice":"1 sentence about daily consumption impact","advice":"2-3 sentence fun personalized health reaction"}}

Verdict guide: excellent=very healthy, good=mostly healthy, moderate=okay occasionally, poor=limit, avoid=very high sugar/fat/sodium.

User profile: {health_profile(user)}
Today so far: {log_str}

Advice/summary rules: use their first name, be clingy if diabetic and food is high sugar/carbs, call out if over daily calories, encourage healthy choices."""

async def groq_vision(img_bytes: bytes, mime: str, prompt: str) -> dict:
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    for model in VISION_MODELS:
        try:
            resp = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=0,
                max_tokens=512,
            )
            return parse_json(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                logger.warning("%s rate limited, waiting 10s...", model)
                await asyncio.sleep(10)
                continue
            logger.warning("%s failed: %s — trying next model", model, err)
            continue
    raise ValueError("All vision models failed. Try again in a moment.")

async def analyze_photo(img_bytes, mime, user, log_str):
    return await groq_vision(img_bytes, mime, build_photo_prompt(user, log_str))

async def analyze_food_qty(img_bytes, mime, user, log_str, qty):
    return await groq_vision(img_bytes, mime, build_food_qty_prompt(user, log_str, qty))

async def groq_casual(user, msg, log_str):
    system = (f"You are NutriScan, a fun caring slightly clingy nutrition assistant on Telegram. "
              f"User profile: {health_profile(user)} | Today: {log_str} | "
              f"Be warm, use their first name occasionally. If they say hi ask what they are eating. "
              f"Be clingy/dramatic about junk food if they have diabetes. 2-3 sentences max.")
    resp = await groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system}, {"role":"user","content":msg}],
        max_tokens=150, temperature=0.8
    )
    return resp.choices[0].message.content.strip()

# ── Onboarding ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)
    user = get_user(chat_id)
    if user and user.get("onboarded"):
        await update.message.reply_text(
            f"Hey {user.get('name')}! 👋 Welcome back!\nSend me a food photo to track today 📸\n\n/today /log /undo /clear /goal /profile")
        return ConversationHandler.END
    await update.message.reply_text(
        "👋 *Welcome to NutriScan v2.0!*\n\nI give nutrition advice tailored *just for you* 🥗\n\nWhat's your *name*? 😊",
        parse_mode="Markdown")
    return ASK_NAME

async def ob_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["ob_name"] = update.message.text.strip().split()[0]
    await update.message.reply_text(f"Love that, {ctx.user_data['ob_name']}! 🎉\nHow old are you? _(number only)_", parse_mode="Markdown")
    return ASK_AGE

async def ob_age(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    t = update.message.text.strip()
    if not t.isdigit() or not (5 <= int(t) <= 120):
        await update.message.reply_text("Enter your age as a number (e.g. 24):"); return ASK_AGE
    ctx.user_data["ob_age"] = int(t)
    await update.message.reply_text("Your *height* in cm? _(e.g. 170)_", parse_mode="Markdown")
    return ASK_HEIGHT

async def ob_height(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: h = float(update.message.text.strip().replace("cm","").strip()); assert 50 <= h <= 280
    except: await update.message.reply_text("Valid height in cm please (e.g. 170):"); return ASK_HEIGHT
    ctx.user_data["ob_height"] = h
    await update.message.reply_text("Your *weight* in kg? _(e.g. 68)_", parse_mode="Markdown")
    return ASK_WEIGHT

async def ob_weight(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: w = float(update.message.text.strip().replace("kg","").strip()); assert 10 <= w <= 500
    except: await update.message.reply_text("Valid weight in kg please (e.g. 68):"); return ASK_WEIGHT
    ctx.user_data["ob_weight"] = w
    bmi, is_obese = compute_bmi(ctx.user_data["ob_height"], w)
    ctx.user_data["ob_bmi"] = bmi; ctx.user_data["ob_is_obese"] = is_obese
    obese_msg = " — I will keep that in mind 💪" if is_obese else " — looking good! 😊"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("No diabetes 🙌", callback_data="diab_none"),
         InlineKeyboardButton("Type 1", callback_data="diab_type1")],
        [InlineKeyboardButton("Type 2", callback_data="diab_type2"),
         InlineKeyboardButton("Pre-diabetes", callback_data="diab_prediabetes")]])
    await update.message.reply_text(
        f"Got it! _(BMI: {bmi}{obese_msg})_\n\nDo you have diabetes?",
        parse_mode="Markdown", reply_markup=kb)
    return ASK_DIABETES

async def ob_diabetes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    diab = q.data.replace("diab_",""); ctx.user_data["ob_diabetes"] = diab
    if diab == "none":
        await _finish_onboarding(q.message, ctx, q.message.chat_id)
        return ConversationHandler.END
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Mild", callback_data="dlevel_mild"),
        InlineKeyboardButton("Moderate", callback_data="dlevel_moderate"),
        InlineKeyboardButton("Severe", callback_data="dlevel_severe")]])
    await q.edit_message_text("What's your diabetes level?", reply_markup=kb)
    return ASK_DIABETES_LEVEL

async def ob_diabetes_level(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    ctx.user_data["ob_diabetes_level"] = q.data.replace("dlevel_","")
    await _finish_onboarding(q.message, ctx, q.message.chat_id)
    return ConversationHandler.END

async def _finish_onboarding(message, ctx, chat_id):
    ob = ctx.user_data
    bmi, is_obese = ob.get("ob_bmi"), ob.get("ob_is_obese", False)
    diab, dlevel = ob.get("ob_diabetes","none"), ob.get("ob_diabetes_level")
    goal = 1600 if is_obese else 2000
    if diab in ("type1","type2"): goal = min(goal, 1800)
    update_user(chat_id, name=ob.get("ob_name"), age=ob.get("ob_age"),
        height_cm=ob.get("ob_height"), weight_kg=ob.get("ob_weight"),
        bmi=bmi, is_obese=int(is_obese), diabetes=diab,
        diabetes_level=dlevel, daily_goal=goal, onboarded=1)
    ctx.user_data.clear()
    u = get_user(chat_id)
    extras = ("\n🩺 I will watch your sugar intake carefully!" if diab != "none" else "") + \
             ("\n⚖️ I will help you stay in a calorie deficit!" if is_obese else "")
    await message.reply_text(
        f"✅ *All set, {u.get('name')}!*\n\n"
        f"• Age: {ob.get('ob_age')} | H: {ob.get('ob_height')}cm | W: {ob.get('ob_weight')}kg | BMI: {bmi}\n"
        f"• Diabetes: {diab.title() if diab!='none' else 'None'}{' ('+dlevel+')' if dlevel else ''}\n"
        f"• Daily Goal: {goal} kcal{extras}\n\n"
        f"Now send me a food photo or nutrition label! 📸🍱",
        parse_mode="Markdown")

# ── Photo flow ──────────────────────────────────────────────────────────────────
async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = get_user(chat_id)
    if not user or not user.get("onboarded"):
        await update.message.reply_text("Set up your profile first with /start 😊")
        return ConversationHandler.END

    msg = await update.message.reply_text("🔍 Scanning your photo...")
    photo = update.message.photo[-1]
    f = await ctx.bot.get_file(photo.file_id)
    buf = BytesIO(); await f.download_to_memory(buf)
    img_bytes = buf.getvalue(); mime = "image/jpeg"

    log_str = today_log_str(chat_id)
    try:
        data = await analyze_photo(img_bytes, mime, user, log_str)
    except Exception as e:
        logger.error("photo analysis failed: %s", e)
        await msg.edit_text("❌ Could not analyse the photo. Please try again with a clearer image!")
        return ConversationHandler.END

    itype = data.get("type", "FOOD")
    advice = data.pop("advice", "")

    if itype == "LABEL":
        ctx.user_data.update({"pending_data": data, "pending_type": "LABEL",
                              "pending_qty": "whole packet", "pending_advice": advice})
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Log this", callback_data="log_confirm"),
            InlineKeyboardButton("❌ Discard",  callback_data="discard")]])
        await msg.edit_text(fmt_result(data) + f"\n\n💬 _{advice}_", parse_mode="Markdown", reply_markup=kb)
        return ConversationHandler.END
    else:
        # Store image for later when user gives quantity
        ctx.user_data.update({"image_bytes": img_bytes, "mime": mime, "pending_type": "FOOD"})
        food_name = data.get("food_name", "this food")
        await msg.edit_text(
            f"🍽️ *{food_name}* detected!\n\n"
            f"{advice if advice else 'How much did you eat?'}\n\n"
            f"_(e.g. '1 cup', '2 chapatis', '200g', 'half plate')_",
            parse_mode="Markdown")
        return WAITING_QUANTITY

async def handle_quantity(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    qty = update.message.text.strip()
    if not qty:
        await update.message.reply_text("Tell me the quantity 😊"); return WAITING_QUANTITY

    chat_id = update.effective_chat.id; user = get_user(chat_id)
    img_bytes = ctx.user_data.get("image_bytes"); mime = ctx.user_data.get("mime","image/jpeg")

    msg = await update.message.reply_text(f"⚖️ Calculating for _{qty}_...", parse_mode="Markdown")
    log_str = today_log_str(chat_id)
    try:
        data = await analyze_food_qty(img_bytes, mime, user, log_str, qty)
    except Exception as e:
        logger.error("food qty failed: %s", e)
        await msg.edit_text("❌ Could not estimate. Please try again!")
        return ConversationHandler.END

    advice = data.pop("advice", "")
    ctx.user_data.update({"pending_data": data, "pending_qty": qty, "pending_advice": advice})
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Log this", callback_data="log_confirm"),
        InlineKeyboardButton("❌ Discard",  callback_data="discard")]])
    await msg.edit_text(fmt_result(data, qty) + f"\n\n💬 _{advice}_", parse_mode="Markdown", reply_markup=kb)
    return ConversationHandler.END

async def cb_log(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer(); chat_id = q.message.chat_id
    if q.data == "discard":
        ctx.user_data.clear(); await q.edit_message_text("🗑️ Discarded!"); return
    data   = ctx.user_data.get("pending_data", {})
    itype  = ctx.user_data.get("pending_type", "FOOD")
    qty    = ctx.user_data.get("pending_qty", "?")
    advice = ctx.user_data.get("pending_advice", "")
    add_entry(chat_id, data, itype, qty, advice)
    ctx.user_data.clear()
    await q.edit_message_text(f"✅ *Logged!*\n\n{fmt_summary(chat_id, get_user(chat_id))}", parse_mode="Markdown")

# ── Casual text ─────────────────────────────────────────────────────────────────
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = get_user(chat_id)
    if not user or not user.get("onboarded"):
        await update.message.reply_text("Run /start to set up your profile first 😊"); return
    try:
        reply = await groq_casual(user, update.message.text.strip(), today_log_str(chat_id))
    except Exception as e:
        logger.error("casual chat failed: %s", e)
        reply = f"Hey {user.get('name')}! 😊 Send me a food photo and I will track it for you!"
    await update.message.reply_text(reply)

# ── Commands ────────────────────────────────────────────────────────────────────
async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = get_user(chat_id)
    if not user or not user.get("onboarded"): await update.message.reply_text("Run /start first!"); return
    await update.message.reply_text(fmt_summary(chat_id, user), parse_mode="Markdown")

async def cmd_log(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = get_user(chat_id); entries = get_today_log(chat_id)
    if not entries:
        await update.message.reply_text(f"📭 Nothing logged yet{', '+user.get('name') if user else ''}! Send a food photo 📸"); return
    lines = [f"📋 *Food Log — {date.today().strftime('%d %b %Y')}*",""]
    for i,e in enumerate(entries,1):
        t = e["logged_at"][11:16] if e["logged_at"] else "?"
        lines.append(f"{i}. {'🏷️' if e['image_type']=='LABEL' else '🍽️'} *{e['food_name']}*\n    `{e['calories']:.0f} kcal` | {e['quantity']} | _{t}_")
    tot = get_today_totals(chat_id)
    lines += ["","─────────", f"🔥 *Total: {tot.get('calories',0):.0f} kcal*"]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_undo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = get_user(chat_id)
    if delete_last(chat_id):
        await update.message.reply_text("↩️ Last entry removed!\n\n" + fmt_summary(chat_id, user), parse_mode="Markdown")
    else: await update.message.reply_text("Nothing to undo — log is empty!")

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Yes clear", callback_data="clear_yes"),
        InlineKeyboardButton("❌ Cancel",    callback_data="clear_no")]])
    await update.message.reply_text("⚠️ Clear today's log?", reply_markup=kb)

async def cb_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if q.data == "clear_yes": clear_today(q.message.chat_id); await q.edit_message_text("🗑️ Cleared!")
    else: await q.edit_message_text("✅ Cancelled!")

async def cmd_goal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = get_user(chat_id); args = ctx.args
    if not args or not args[0].isdigit():
        await update.message.reply_text(f"🎯 Goal: *{user.get('daily_goal',DEFAULT_GOAL)} kcal/day*\nChange: `/goal 1800`", parse_mode="Markdown"); return
    g = int(args[0])
    if not (500 <= g <= 10000): await update.message.reply_text("Must be 500–10000 kcal."); return
    update_user(chat_id, daily_goal=g)
    await update.message.reply_text(f"✅ Goal set to *{g} kcal*! 🎯", parse_mode="Markdown")

async def cmd_profile(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; u = get_user(chat_id)
    if not u or not u.get("onboarded"): await update.message.reply_text("No profile yet! Run /start."); return
    diab = u.get("diabetes","none"); dl = u.get("diabetes_level")
    await update.message.reply_text(
        f"👤 *Your Profile*\n\n"
        f"• Name: {u.get('name')}\n• Age: {u.get('age')} yrs\n"
        f"• H: {u.get('height_cm')}cm | W: {u.get('weight_kg')}kg | BMI: {u.get('bmi')} "
        f"{'⚠️ Obese' if u.get('is_obese') else '✅ Normal'}\n"
        f"• Diabetes: {'None' if diab=='none' else diab.title()+(f' ({dl})' if dl else '')}\n"
        f"• Daily Goal: {u.get('daily_goal')} kcal\n\n_Run /start to update._",
        parse_mode="Markdown")

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    onboarding = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            ASK_NAME:           [MessageHandler(filters.TEXT & ~filters.COMMAND, ob_name)],
            ASK_AGE:            [MessageHandler(filters.TEXT & ~filters.COMMAND, ob_age)],
            ASK_HEIGHT:         [MessageHandler(filters.TEXT & ~filters.COMMAND, ob_height)],
            ASK_WEIGHT:         [MessageHandler(filters.TEXT & ~filters.COMMAND, ob_weight)],
            ASK_DIABETES:       [CallbackQueryHandler(ob_diabetes, pattern="^diab_")],
            ASK_DIABETES_LEVEL: [CallbackQueryHandler(ob_diabetes_level, pattern="^dlevel_")],
        },
        fallbacks=[CommandHandler("cancel", lambda u,c: ConversationHandler.END)],
        per_user=True, per_chat=True)

    photo_conv = ConversationHandler(
        entry_points=[MessageHandler(filters.PHOTO, handle_photo)],
        states={WAITING_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)]},
        fallbacks=[CommandHandler("cancel", lambda u,c: ConversationHandler.END)],
        per_user=True, per_chat=True)

    app.add_handler(onboarding)
    app.add_handler(photo_conv)
    app.add_handler(CommandHandler("today",   cmd_today))
    app.add_handler(CommandHandler("log",     cmd_log))
    app.add_handler(CommandHandler("undo",    cmd_undo))
    app.add_handler(CommandHandler("clear",   cmd_clear))
    app.add_handler(CommandHandler("goal",    cmd_goal))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CallbackQueryHandler(cb_clear, pattern="^clear_"))
    app.add_handler(CallbackQueryHandler(cb_log,   pattern="^(log_confirm|discard)$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("NutriScan v2.0 started — fully on Groq")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()