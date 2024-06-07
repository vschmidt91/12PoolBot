import os
import requests

API_TOKEN_ENV: str = "AI_ARENA_API_TOKEN"
BOT_ID_ENV: str = "AI_ARENA_BOT_ID"
ZIPFILE_NAME: str = "bot.zip"

token = os.environ.get(API_TOKEN_ENV)
bot_id = os.environ.get(BOT_ID_ENV)
print(bot_id, token)
url = f'https://aiarena.net/api/bots/{bot_id}/'
with open(ZIPFILE_NAME, "rb") as bot_zip:
    req_data = {
        "bot_zip_publicly_downloadable": True,
        # "bot_data_publicly_downloadable": False,
        # "bot_data_enabled": True,
        # "wiki_article_content": descriptions[name],
    }
    req_files = {
        "bot_zip": bot_zip
    }
    r = requests.patch(url, headers={"Authorization": f"Token {token}"}, data=req_data, files=req_files)
    print(r)
    print(r.content)
