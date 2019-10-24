import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url_breeds = 'https://www.googleapis.com/drive/v3/files/13QpJ7isrNTUL2GWIFU9Sq4ectXQevunx?alt=media&key=AIzaSyDxkCFTSW6M8CIJgOKVy8ANkD2ceHvyo1s'
export_file_name_breeds = 'breeds.pkl'

export_file_url_other = 'https://www.googleapis.com/drive/v3/files/1jriuptG7twsH8dVay_HlJ31lJxkwbCDR?alt=media&key=AIzaSyDxkCFTSW6M8CIJgOKVy8ANkD2ceHvyo1s'
export_file_name_other = 'other.pkl'

classes_breeds = ['Giant Schnazer','Black Russian Terrier']
classes_other = ['Other','GS or BRT']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url_breeds, path / export_file_name_breeds)
    await download_file(export_file_url_other, path / export_file_name_other)
    try:
        learn_breeds = load_learner(path, export_file_name_breeds)
        learn_other = load_learner(path, export_file_name_other)
        return learn_breeds, learn_other
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn_breeds, learn_other = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    if str(learn_other.predict(img)[0]) == 'Other':
        return JSONResponse({'result': 'Не похоже на чёрного терьера или ризеншнауцера.'})    
    else:
        prediction = learn_breeds.predict(img)[0]
        return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
