import os
import sys
import logging
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv
import google.generativeai as genai

# discord_only_plus_gemini.py
# Discord bot that integrates with Google Gemini AI for intelligent responses
# Requires: discord.py, google-generativeai, python-dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokens from environment
DISCORD_TOKEN = os.getenv("TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN:
    logger.error("Environment variable TOKEN not set. Exiting.")
    sys.exit(1)

if not GEMINI_API_KEY:
    logger.error("Environment variable GEMINI_API_KEY not set. Exiting.")
    sys.exit(1)

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Load system prompt from external file
SYSTEM_PROMPT_FILE = "system_prompt.txt"
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
    logger.info("System prompt loaded from %s", SYSTEM_PROMPT_FILE)
except FileNotFoundError:
    logger.error("System prompt file '%s' not found. Exiting.", SYSTEM_PROMPT_FILE)
    sys.exit(1)
except Exception as e:
    logger.error("Error reading system prompt file: %s. Exiting.", e)
    sys.exit(1)

# Create Gemini model with system instruction
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction=SYSTEM_PROMPT,
    generation_config={
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40,
        'max_output_tokens': 8192,
    }
)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # required to read message content

bot = commands.Bot(command_prefix="$question ", intents=intents, help_command=None, case_insensitive=True)


@bot.event
async def on_ready():
    logger.info("Bot logged in as %s (id=%s)", bot.user, bot.user.id)
    guilds = ", ".join(g.name for g in bot.guilds) or "no guilds"
    logger.info("Connected to: %s", guilds)
    logger.info("Gemini AI integration ready!")


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if message starts with $question
    if message.content.startswith("$question "):
        # Extract the question (remove the command prefix)
        question = message.content[len("$question "):].strip()

        if not question:
            await message.channel.send("Please ask a question after `$question`!")
            return

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                # Create chat (system instruction is already in the model)
                chat = model.start_chat(history=[])

                # Send user question directly (system instruction is already configured)
                # Use asyncio.to_thread to avoid blocking the event loop
                response = await asyncio.to_thread(
                    chat.send_message,
                    question,
                    safety_settings={
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )

                # Check if response was blocked
                if not response.parts:
                    # Log the finish reason
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown"
                    safety_ratings = response.candidates[0].safety_ratings if response.candidates else []

                    logger.warning(
                        "Response blocked - finish_reason: %s, safety_ratings: %s",
                        finish_reason,
                        safety_ratings
                    )

                    await message.channel.send(
                        "Sorry, I couldn't generate a response. This might be due to content filters. "
                        "Please try rephrasing your question."
                    )
                    return

                # Get the response text
                response_text = response.text

                # Discord has a 2000 character limit, split intelligently if needed
                if len(response_text) > 2000:
                    # Split into chunks at natural break points (newlines, periods, spaces)
                    chunks = []
                    current_chunk = ""

                    # Split by paragraphs first (double newlines)
                    paragraphs = response_text.split('\n\n')

                    for para in paragraphs:
                        # If adding this paragraph would exceed limit
                        if len(current_chunk) + len(para) + 2 > 2000:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""

                            # If single paragraph is too long, split by sentences
                            if len(para) > 2000:
                                sentences = para.replace('. ', '.\n').split('\n')
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) + 1 > 2000:
                                        if current_chunk:
                                            chunks.append(current_chunk.strip())
                                        current_chunk = sentence + " "
                                    else:
                                        current_chunk += sentence + " "
                            else:
                                current_chunk = para + "\n\n"
                        else:
                            current_chunk += para + "\n\n"

                    # Add remaining chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    # Send all chunks
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response_text)

                logger.info("Responded to question from %s: %s", message.author, question[:50])

            except Exception as e:
                logger.exception("Error processing Gemini request: %s", e)
                await message.channel.send(f"Sorry, I encountered an error: {str(e)}")

    # Process other commands (if any are added)
    await bot.process_commands(message)


@bot.command(name="help", help="Shows how to use the bot")
async def help_command(ctx: commands.Context):
    help_text = """**Discord + Gemini AI Bot**

Use `$question <your question>` to ask me anything!

Examples:
- `$question What is Python?`
- `$question Explain quantum computing`
- `$question Tell me a joke`

I'm powered by Google Gemini AI and ready to help!"""
    await ctx.send(help_text)


def main():
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    except Exception as e:
        logger.exception("Bot crashed: %s", e)


if __name__ == "__main__":
    main()
