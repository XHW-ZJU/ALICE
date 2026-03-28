import pandas
from image_quant import in_vitro_quantification
import skimage
import pandas as pd
import os
import datetime

def is_fig(name: str):
    return name[-3:].lower() == 'jpg' or name[-3:].lower() == 'tif'

def quantification(photo_path: str):
    file_list = os.listdir(photo_path+r'\red')
    file_list = list(filter(is_fig, file_list))
    print(file_list)
    df = pd.DataFrame(columns=['total_area', 'signa_area', 'to_sg_area'])
    for _, file in enumerate(file_list):
        path_sig = photo_path+r'\red\{}'.format(file)
        path_bf = photo_path+r'\green\{}'.format(file)
        im_sig = skimage.io.imread(path_sig)
        if len(im_sig.shape) == 3:
            im_sig = skimage.img_as_float(im_sig[:, :, 0])
        elif len(im_sig.shape) == 1:
            im_sig = skimage.img_as_float(im_sig)
        else:
            im_sig = skimage.img_as_float(im_sig)
        im_bf = skimage.img_as_float(skimage.io.imread(path_bf)[:, :, 1])
        print("quantifying_fig:{}".format(file))
        res = in_vitro_quantification(im_bf, im_sig, photo_name=file)
        dict_for_df = {
            'quant_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'file_name_': file,
            'total_area': res[0],
            'signa_area': res[1],
            'to_sg_area': res[2]
        }
        df_temp = pandas.DataFrame.from_dict([dict_for_df])
        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_excel(photo_path+r'\res.xlsx')


