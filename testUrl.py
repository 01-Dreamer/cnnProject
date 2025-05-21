from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
from cnnModel import CNN


# Flask框架初始化
app = Flask(__name__, template_folder='.')
device = torch.device("cpu")
model = CNN().to(device)
try:
    model.load_state_dict(torch.load('digit_cnn.pth', map_location=device, weights_only=True))
    print("模型加载成功")
except Exception as e:
    print(f"加载模型出错: {e}")
    exit(1)
model.eval()


# 预处理图片
def preprocess_image(image):
    try:
        image = image.convert('L')
        image = image.resize((28, 28))
        image.save("static/debug_processed_image.png")
        print("预处理图片已保存 'static/debug_processed_image.png'")

        img_array = np.array(image, dtype=np.uint8)
        img_array = 255 - img_array

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_array).unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"预处理图片出错: {e}")
        return None


# 预测数字
def predict_digit(image_tensor):
    try:
        if image_tensor is None:
            return "错误: 无效的图片张量"
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return f"预测数字: {prediction}"
    except Exception as e:
        return f"预测过程错误: {str(e)}"


# 主页路由
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    original_image = None
    debug_image = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result = "错误: 未选择文件"
        else:
            file = request.files['image']
            if file.filename == '':
                result = "错误: 未选择有效文件"
            else:
                try:
                    image = Image.open(file)
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    original_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_tensor = preprocess_image(image)
                    result = predict_digit(img_tensor)
                    debug_image = "/static/debug_processed_image.png"
                except Exception as e:
                    result = f"处理图片出错: {str(e)}"

    return render_template('index.html', result=result, original_image=original_image, debug_image=debug_image)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)
