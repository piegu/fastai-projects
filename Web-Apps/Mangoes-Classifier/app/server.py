from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1j_olmDIgVRzzCAszgYwGb_ldc6tdekI0'
export_file_name = 'mangoes-classifier-resnet34-export.pkl'

classes = ['Palmer', 'Tommy Atkins']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    # html = open(html, encoding="utf-8")
    # return HTMLResponse(html.read())
    return HTMLResponse(html.open().read().decode("utf8"))

# @app.route('/analyze', methods=['POST'])
# async def analyze(request):
#     data = await request.form()
#     img_bytes = await (data['file'].read())
#     img = open_image(BytesIO(img_bytes))
#     prediction = learn.predict(img)[0]
#     return JSONResponse({'result': str(prediction)})


@app.route('/analyze', methods=['POST'])
async def analyze(request):

    # label prediction function 
    # to get a prediction with 90% of confidence in each class, it is necessary to setup distinct thresholds
    # Our classifier detects with more facility class 0 than class 1
    def predict_label(img, ths=[0.8535353535353536, 0.6414141414141414], model=learn):
        
        # Condition function
        def cond(idx):
            return (indice == idx) and (pred >= ths[idx])

        # Get prediction values
        cat, indice, preds = model.predict(img)
        pred, _ = torch.max(preds, 0)
        indice = indice.item()
        pred = pred.item()
        preds = list(preds)

        # Check the confidence of the classifier in its prediction
        if cond(0) or cond(1):
            prediction = cat
        else:
            prediction = 'unrecognized'
            indice = 2
        
        return prediction, indice, pred, preds

    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction, indice, pred, preds = predict_label(img)
    prob = f'{np.round(100*pred,2)}%'
    probs = [f'{np.round(100*preds[i].item(),2)}%' for i in range(2)]
    probs = [f'Palmer ({probs[0]})', f'Tommy Atkins ({probs[1]})']
    return JSONResponse({'prediction': str(prediction), 'class': indice, 'prob': prob, 'probs': probs})


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
