from model import StyleTransferModel
from telegram_token import token
import config
import time
from telegram.ext.dispatcher import run_async
from threading import Lock

from io import BytesIO

# TODO:
# 6) пофиксить ошибки pyCharm'


model = StyleTransferModel()
first_image_file = {}
style_transfer_mutex = Lock()  # we use only one model for all images


def send_msg(update, context, text):
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)


@run_async
def send_prediction_on_photo_async(update, context, content_image_file, style_image_file):
    send_msg(update, context, "Please, wait. I'm working...")

    content_image_stream = BytesIO()
    content_image_file.download(out=content_image_stream)

    style_image_stream = BytesIO()
    style_image_file.download(out=style_image_stream)

    with style_transfer_mutex:
        start_time = time.clock()
        output = model.transfer_style(content_image_stream, style_image_stream)
        print("Elapsed time:", time.clock() - start_time)

    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    context.bot.send_photo(update.message.chat_id, photo=output_stream)
    send_msg(update, context, "Done! Please, send me new image to stylize!")
    print("Sent Photo to user")


def send_prediction_on_photo(update, context):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    image_info = update.message.photo[-1]
    image_file = context.bot.get_file(image_info)
    
    if chat_id in first_image_file:
        content_image_file = first_image_file[chat_id]
        style_image_file = image_file
        del first_image_file[chat_id]

        send_prediction_on_photo_async(update, context, content_image_file, style_image_file)
    else:
        first_image_file[chat_id] = image_file
        send_msg(update, context, "Please, send image with new style.")


def start(update, context):
    send_msg(update, context, "I'm a style transfer bot, please, send me image to stylize!")


def reset(update, context):
    try:
        del first_image_file[update.message.chat_id]
        send_msg(update, context, "Your old image is forgotten! Please, send me new image to stylize!")
    except KeyError:
        send_msg(update, context, "You didn't send any image :-(")


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
    import logging
    
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    if config.use_proxy:
        updater = Updater(token=token, use_context=True, request_kwargs={'proxy_url': config.proxy_url})
    else:
        updater = Updater(token=token, use_context=True)

    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    reset_handler = CommandHandler('reset', reset)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(reset_handler)
    dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
    updater.start_polling()
    updater.idle()
