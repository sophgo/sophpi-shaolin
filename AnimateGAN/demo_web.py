import streamlit as st 
from PIL import Image
import numpy as np 
from tools import load_test_data_from_pil, get_save_images
from engine import EngineOV

def run(model):
    st.header("Sophon AI ZOOM")
    st.header("图像风格转换：动漫风格")
    st.write('本demo演示了如何使用sophon AI ZOOM平台进行图像风格转换，将输入的图片转换为动漫风格(用于场景转换，而非人物转换）。')
    #open the input file
    init_image = st.file_uploader("source_image", type=['jpg','png','jpeg'])
    if init_image:
        # 获取输入图片 
        img = Image.open(init_image)
        st.image(img, caption='input image', use_column_width=True)

        # 转为opencv 
        img = np.array(img)
        # rgb -> bgr 
        image, source_size = load_test_data_from_pil(img, (1024,700))
        sample_image = np.asarray(image)
        res = model({"test": sample_image})
        ret_img = get_save_images(res,size=source_size[::-1], source=img)
        
        # 显示结果
        st.image(ret_img, caption='output image', use_column_width=True)

if __name__ == '__main__':
    bmodel_path = './animategan.bmodel'
    model = EngineOV(bmodel_path, 3)
    run(model)


