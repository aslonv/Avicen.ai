from telegram.ext import Application, CommandHandler, MessageHandler, filters
from llama_service import LlamaHealthService

async def start(update, context):
    await update.message.reply_text("Welcome to Avicen Health! Share your health data for analysis.")

async def handle_message(update, context):
    text = update.message.text
    llama_service = LlamaHealthService(
        model_path="path/to/llama-model",
        weather_api_key="your_key"
    )
    response = llama_service.generate_response(text)
    await update.message.reply_text(response)

def main():
    app = Application.builder().token("7737349429:AAHcuIngs3ba_bdZoRRB6j05wQSG_Iy8d8A").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()