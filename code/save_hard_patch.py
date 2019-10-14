# -*- coding: utf-8 -*-
import sys
import os
import io
import cv2
import glob
import json
import pickle
import pdb
import numpy as np
from count_node import count_lymph, tranfer_id
from ReadHeatmap_old import heatmap_visualize
from blend import heatmap_shape_adjust
from PIL import Image
# from kf_tmap_slide import open_slide
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from mask_generator.mask_generator import MaskOnTheFly
import sys
import math

styleTypes = [{"color":"#fa0000","id":"ff80818166dc26da0166e29bbba10147","name":"癌组织"}]

def coordinate_tansfer(contours):
    contours = np.array(contours)
    level_7_contours = contours/2**7

    return level_7_contours

def extract_foreground_mask(img, threshold=0.75, dilate_kernel=2):
    """
    Func: Get a gray image from slide
    
    Args: img

    Returns:gray_t

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    # Convert color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray_t = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY_INV)
    gray_t = cv2.dilate(gray_t, kernel)
    ret, gray_t = cv2.threshold(gray_t, threshold * 255, 255, cv2.THRESH_BINARY)
    return gray_t


def extract_fg_patch(fgmask_image, x_cut_id, y_cut_id, cut_patch_size_w, cut_patch_size_h, level, level_fgmask=3):
    """
    Func: cut fg-mask patch from fg-mask image

    Args: (fgmask_image, x_cut_id, y_cut_id, cut_patch_size_w, cut_patch_size_h, level, level_fgmask)

    level_fgmask: default level-2

    Returns: 
        patch_fg: numpy array
    """
    x_cut_id_fgmask = x_cut_id // (2 ** (level_fgmask - level))
    y_cut_id_fgmask = y_cut_id // (2 ** (level_fgmask - level))
    cut_patch_x_fgmask = cut_patch_size_w // (2 ** (level_fgmask - level)) + x_cut_id_fgmask
    cut_patch_y_fgmask = cut_patch_size_h // (2 ** (level_fgmask - level)) + y_cut_id_fgmask
    fgmask_patch = fgmask_image[x_cut_id_fgmask:cut_patch_x_fgmask, y_cut_id_fgmask:cut_patch_y_fgmask]

    patch_fg = cv2.resize(fgmask_patch, (cut_patch_size_w, cut_patch_size_h), interpolation=cv2.INTER_AREA) # default a bilinear interpolation

    return patch_fg


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = 1000000000
    # pkl_base_path = '/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/train_pkl/8_29_one_wsi/tumor'
    # pkl_base_path = "/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/train_pkl/7_30_epoch_8/tumor"
    # wsi_base_path = '/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/zhang_train_tumor/'
    normal_wsi_base_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/WSI_XML/zhang_train_normal/train_normal/"
    normal_pkl_base_path = "/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/train_pkl/5_fgmask_all_normal/normal/"
    # json_base_path = '/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/normal_wrong_patch/5200_normal_patch_epoch_8/contours_json'

    # tumor_wsi_json_base_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/zhang_train_json/"
    # yaml_file_path = "/mnt/disk_share/yufeng/train_code_Camylon/mask_info.yaml"
    heatmap_level = 7
    patch_size = 1792
    thres = 0.50
    label = "normal"
    # label = "tumor"
    normal_fgmask_dir = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/normal_fg_mask_level3"
    # fgmask_dir = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/tumor_fg_total_wsi_level_3"

    heatmap_pkl_paths = glob.glob(os.path.join(normal_pkl_base_path, '*.pkl'))
    heatmap_pkl_paths.sort()
    # heatmap_pkl_paths = heatmap_pkl_paths[72:]
    # heatmap_pkl_paths = heatmap_pkl_paths
    # save_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/test_wsi_xml_one/save/tumor/"
    # save_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/hard_tumor/"
    save_path = "/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/train_pkl/5_fgmask_all_normal/hard_normal/"
    num = 0 # 运行一次就要加一次
    print('the number of heatmap is:', len(heatmap_pkl_paths))

    # wsi_paths = glob.glob(os.path.join(wsi_base_path, '*.TMAP'))
    wsi_paths = glob.glob(os.path.join(normal_wsi_base_path, '*.ndpi'))
    # print('the number of wsi is:', len(wsi_paths))
    wsi_paths.sort()
    # wsi_paths = wsi_paths[72:]
    print('the number of wsi is:', len(wsi_paths))
    # wsi_paths = wsi_paths
    # json_paths = glob.glob(os.path.join(tumor_wsi_json_base_path, '*.json'))
    # print('the number of wsi_json_file is:', len(json_paths))
    # json_paths.sort()
    # json_paths = json_paths[7:]
    # pdb.set_trace()
    # mask_gen = MaskOnTheFly(json_path, yaml_file_path)
    a = 0
    # # =================debug==========================
    # wsi_paths = ["/mnt/disk_share/dulicui/project/colorectal/dataset/V4/test_colon_biopsy_20190605/B099348101_C_1_20190227162908.TMAP"]
    # heatmap_pkl_paths = ["./heatmap_test_colon_biopsy_20190605/heatmap_pkl/B099348101_C_1_20190227162908_tumor.pkl"]
    # print("wsi len and pkl len:", len(wsi_paths), len(heatmap_pkl_paths))
    # # =================debug===============================
    # pdb.set_trace()
    # for wsi_path, pkl_path,json_path in zip(wsi_paths, heatmap_pkl_paths,json_paths):
    for wsi_path, pkl_path in zip(wsi_paths, heatmap_pkl_paths):
    # wsi_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/zhang_train_tumor/823029-9 - 2018-09-18 11.34.46.ndpi"
    # pkl_path = "/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/train_pkl/8_27_test/tumor/train_823029-9 - 2018-09-18 11.34.46.ndpi_tumor.pkl"
    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/zhang_train_json/823029-9 - 2018-09-18 11.34.46.json"
        # mask_gen = MaskOnTheFly(json_path, yaml_file_path)
        meta_data_array = {"image_id": "unknow",
            "data_origin": "unknow",
            "level": "0",
            "label": "normal",
            "patches": []
        }

        meta_data_array["image_id"] = os.path.basename(wsi_path)
        meta_data_array["data_origin"] = wsi_path
        print('INFO: ============dealing with %s' % (pkl_path.split('/')[-1]))
        wsi_name = os.path.basename(wsi_path)[:-5]
        anno_dic = {
            "data": [],
            "fromUserId": "",
            "styleTypes": styleTypes
        }

        # pdb.set_trace()
        try:
            # slide = open_slide(wsi_path)
            slide = OpenSlide(wsi_path)
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            pass
        slide_w, slide_h = slide.level_dimensions[0]


        heat_map_img_224 = pickle.load(open(pkl_path, 'rb'))
        heat_map_vis = heatmap_visualize(file_name=heat_map_img_224, heat_map=heat_map_img_224, threshold=0.5)
        heat_map_img = heatmap_shape_adjust(heat_map_vis, slide_w, slide_h, patch_size, level_output=heatmap_level, network='RESNET_50')
        print("heat_map_img.shape", heat_map_img.shape)
        heat_map_height,heat_map_width = heat_map_img.shape
        ratio_h = slide_h / heat_map_height
        ratio_w = slide_w / heat_map_width

        heat_map_img[heat_map_img<(thres*255)] = 0
        _, contours, _ = cv2.findContours(heat_map_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if label == "normal":
            # patch_size = 800
            contours.sort(key=lambda elem: len(elem))
            contours = contours[::-1]
            if len(contours) > 0:
                index = 0
                for i in range(len(contours)):
                    contour = contours[i:i+1]
                    if len(contours[i][:, 0, :]) == 1:
                        # pdb.set_trace()
                        continue
                    elif max(contours[i][:, 0, :][:, 0]) == min(contours[i][:, 0, :][:, 0]) or max(contours[i][:, 0, :][:, 1]) == min(contours[i][:, 0, :][:, 1]):
                        # pdb.set_trace()
                        print(contours[i][:, 0, :])
                        continue
                    else:
                        # pdb.set_trace()
                        # contours_count.append(contours[i])
                        h, w = heat_map_img.shape
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask, contour, -1, 1, -1)
                        # cv2.imwrite("./picture/mask.png",mask)
                        mask = mask.astype(bool)
                        area = heat_map_img[mask]
                        # cv2.imwrite("./picture/heat_map_img.png",heat_map_img)
                        # max_diameter = max_dia_handle(contour, level)
                        # max_diameters.append(max_diameter)
                        # contour_data["max_diameter"] = max_diameter
                        pred_max = max(area)
                        # contour_data["pred_max"] = round((pred_max/2.55), 2)
                        actual_pred_max = round((pred_max/2.55), 2)
                        index += 1
                        coord_dict = {
                        # "name": "annotation {}" .format(index),
                        "name": "annotation {} {}" .format(index,actual_pred_max),
                        # "name": str(actual_pred_max)
                        "points": "unkonw",
                        "styleType": "ff80818166dc26da0166e29bbba10147",
                        "type": "polygon"
                        }
                        coordinate_level_0 = tranfer_id(wsi_path, output_level=0, heatmap=heat_map_img, contours=contour, contour_level=heatmap_level)
                        coordinate_level_0_all = coordinate_level_0[0, :, 0, :]
                        x = min(coordinate_level_0_all[:,0])
                        y = min(coordinate_level_0_all[:,1])
                        w_w = int(max(coordinate_level_0_all[:,0]) - min(coordinate_level_0_all[:,0]))
                        h_h = int(max(coordinate_level_0_all[:,1]) - min(coordinate_level_0_all[:,1]))

                        cut_patch_size = min(w_w,h_h)

                        if w_w <224:
                            w_w = 224
                        if h_h <224:
                            h_h = 224
                        cut_patch_size = min(w_w,h_h)
                        
                        if cut_patch_size>2000:
                            cut_patch_size = 2000

                        # h_h = max(w_w,h_h)  
                        # target_region = slide.read_region((x, y), 0, (int(w), int(h)))

                        # count_x = 0
                        # count_y = 0
                        for width in range(0,w_w,cut_patch_size):
                            if width + cut_patch_size > w_w:
                                top_left_x = int(w_w - cut_patch_size + x)
                            else:
                                top_left_x = int(width + x) 
                            # count_x +=1
                            for heigth in range(0,h_h,cut_patch_size):
                                if heigth + cut_patch_size > h_h:
                                    top_left_y = int(h_h - cut_patch_size + y) 
                                else:
                                    top_left_y = int(heigth + y) 

                                patch_224 = slide.read_region((top_left_x, top_left_y), 0, (cut_patch_size, cut_patch_size))
                                patch_224_array = np.array(patch_224)
                                patch_image = np.array(patch_224_array)
                                patch_image = patch_image[:, :, 0:3]
                                patch = Image.fromarray(patch_image)
                                # pdb.set_trace()
                                fgmask_path = os.path.join(normal_fgmask_dir, "fgmask_" + meta_data_array["image_id"] + ".png")
                                fgmask_image = np.transpose(np.array(Image.open(fgmask_path)), (1,0))
                                # print(top_left_x)
                                fgmask_path_224 = extract_fg_patch(fgmask_image,top_left_y,top_left_x,cut_patch_size,cut_patch_size,level=0,level_fgmask=3)
                                # fgmask_path_224 = Image.fromarray(fgmask_path_224)
                                fg_img_nonzero_rate = np.count_nonzero(fgmask_path_224)/(cut_patch_size*cut_patch_size)
                                if fg_img_nonzero_rate >= 0.1:
                                    # impo
                                    # save_path = "/mnt/disk_share/yufeng/train_code_Camylon/slide_inference_pipeline/normal_wrong_patch/5200_normal_patch_epoch_8/"
                                    patch_save_path = save_path + "patch/" + "img_" + os.path.basename(wsi_path) + "_level_0" + "_x_" + str(top_left_x) + "_y_" + str(top_left_y) +".png"
                                    patch.save(patch_save_path)

                                    fgmask_save_path = save_path + "fgmask/" + "fgmask_" + os.path.basename(wsi_path) + "_level_0" + "_x_" + str(top_left_x) + "_y_" + str(top_left_y) +".png"
                                    fgmask_path_224 = Image.fromarray(fgmask_path_224)
                                    fgmask_path_224.save(fgmask_save_path)
                                    
                                    meta_dict = {"patch_id": "unknown",
                                    "img_path": "unknown",
                                    "mask_path": "unknown",
                                    "patch_size": [int(patch_size),int(patch_size)],
                                    "fgmask_path": "unknown"
                                    }

                                    # pdb.set_trace()
                                    meta_dict["patch_id"] = [top_left_x,top_left_y]
                                    meta_dict["img_path"] = patch_save_path
                                    meta_dict["mask_path"] = "None"
                                    meta_dict["fgmask_path"] = fgmask_save_path
                                    meta_data_array["patches"].append(meta_dict)
                    
                                    # count_y +=1
                        # coordinate_level_0 = unique(coordinate_level_0_all)
                        coord_dict["points"] = coordinate_level_0_all.tolist() #list(list(coordinate_level_0_all))
                        anno_dic["data"].append(coord_dict)
                print('INFO: %s is done, start to save json' % (wsi_path))
        else:
            meta_data_array = {"image_id": "unknow",
                            "data_origin": "unknow",
                            "level": "0",
                            "label": "tumor",
                            "patches": []
                            }

            meta_data_array["image_id"] = os.path.basename(wsi_path)
            meta_data_array["data_origin"] = wsi_path

            json_open = open(json_path)
            json_array = json.load(json_open)
            # meta_data_array["image_id"] =  str(wsi_path)

            wsi_contour_list_level_7 = []
            heatmap_list_level_7 = []
            wsi_min_x_list = []
            wsi_min_y_list = []
            wsi_max_x_list = []
            wsi_max_y_list = []

            heatmap_min_x_list = []
            heatmap_min_y_list = []
            heatmap_max_x_list = []
            heatmap_max_y_list = []

            all_contours = json_array["data"]
            for one_contours in all_contours:
                json_contours = one_contours["points"]
                # pdb.set_trace()
                level_7_contours = coordinate_tansfer(json_contours)
                # pdb.set_trace()
                wsi_min_x_list.append(min(level_7_contours[:,0]))
                wsi_min_y_list.append(min(level_7_contours[:,1]))
                wsi_max_x_list.append(max(level_7_contours[:,0]))
                wsi_max_y_list.append(max(level_7_contours[:,1]))

                wsi_contour_list_level_7.append(level_7_contours)

            # wsi_contour_list_level_7 = np.array(wsi_contour_list_level_7)
            # pdb.set_trace()
            # wsi_min_x_list = []
            wsi_min_x = min(wsi_min_x_list)
            wsi_min_y = min(wsi_min_y_list)
            wsi_max_x = max(wsi_max_x_list)
            wsi_max_y = max(wsi_max_y_list)

            wsi_w_w = int(wsi_max_x - wsi_min_x)
            wsi_h_h = int(wsi_max_y - wsi_min_y)

            for i in range(len(contours)):
                    contour = contours[i:i+1]
                    if len(contours[i][:, 0, :]) == 1:
                        # pdb.set_trace()
                        continue
                    elif max(contours[i][:, 0, :][:, 0]) == min(contours[i][:, 0, :][:, 0]) or max(contours[i][:, 0, :][:, 1]) == min(contours[i][:, 0, :][:, 1]):
                        # pdb.set_trace()
                        print(contours[i][:, 0, :])
                        continue
                    else:
                        # pdb.set_trace()
                        heatmap_list_level_7.append(np.array(contour[0][:,0,:].tolist()))
                        heatmap_min_x_list.append(min(contour[0][:,0,:][:,0]))
                        heatmap_min_y_list.append(min(contour[0][:,0,:][:,1]))
                        heatmap_max_x_list.append(max(contour[0][:,0,:][:,0]))  
                        heatmap_max_y_list.append(max(contour[0][:,0,:][:,1]))  

            # pdb.set_trace()

            heatmap_list_level_7 = np.array(heatmap_list_level_7)
            heatmap_min_x = min(heatmap_min_x_list)
            heatmap_min_y = min(heatmap_min_y_list)
            heatmap_max_x = max(heatmap_max_x_list)
            heatmap_max_y = max(heatmap_max_y_list)

            heatmap_w_w = int(heatmap_max_x - heatmap_min_x)
            heatmap_h_h = int(heatmap_max_y - heatmap_min_y)
            zero_shape_wsi = np.zeros((heat_map_height,heat_map_width))
            zero_shape_heatmap = np.zeros((heat_map_height,heat_map_width))

            new_wsi_conturs = []
            new_heatmap_contours = []

            for i in range(len(wsi_contour_list_level_7)):
                contour_list_level_7 = wsi_contour_list_level_7[i]
                yy = contour_list_level_7.astype("int32")
                new_wsi_conturs.append(yy)
                
            cv2.drawContours(zero_shape_heatmap,heatmap_list_level_7,-1,(255,255,255),-1)
            cv2.drawContours(zero_shape_wsi,new_wsi_conturs,-1,(255,255,255),-1)

            cv2.imwrite("./wsi.png",zero_shape_wsi)
            cv2.imwrite("./heatmap.png",zero_shape_heatmap)

            # cv2.drawContours(zero_shape_heatmap,new_heatmap_contours,-1,(255,255,255),3)

            result_yu = zero_shape_wsi * zero_shape_heatmap
            cv2.imwrite("./yu.png",result_yu)
            result_yu_1_0 = result_yu.astype(bool).astype(int)

            # zero_shape_wsi_same = result_yu_1_0 * zero_shape_wsi

            zero_shape_wsi_predict = result_yu_1_0 * zero_shape_wsi
            cv2.imwrite("./zero_shape_wsi_predict.png",zero_shape_wsi_predict)
            # cv2.imwrite("./result_yu.png",result_yu)

            tumor_hard_sample = cv2.absdiff(zero_shape_wsi, zero_shape_wsi_predict)
            cv2.imwrite("./tumor_hard_sample.png",tumor_hard_sample)
            # cv2.imwrite("./tumor_hard_sample.png",tumor_hard_sample)
            tumor_hard_sample = tumor_hard_sample.astype("uint8")
            # cv2.imwrite("./look_4.png",tumor_hard_sample)
            # tumor_hard_sample = gray_t.astype("uint8")cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_NONE
            _, tumor_contours, _ = cv2.findContours(tumor_hard_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            zero_shape_heatmap_new = np.zeros((heat_map_height,heat_map_width))
            cv2.drawContours(zero_shape_heatmap_new,tumor_contours,-1,(255,255,255),0)
            cv2.imwrite("./look.png",zero_shape_heatmap_new)

            json_open = open(json_path)
            json_array = json.load(json_open)
            json_array_copy = json_array.copy()
            json_array_copy["data"] = []
            for i in range(len(tumor_contours)):
                dict_ = {"name": "unknown",
                         "points": "unknown",
                         "styleType" : "ff80818166dc26da0166e29bbba10147",
                         "type": "polygon"
                        }
                if len(tumor_contours[i][:, 0, :]) == 1:
                        # pdb.set_trace()
                    continue
                elif max(tumor_contours[i][:, 0, :][:, 0]) == min(tumor_contours[i][:, 0, :][:, 0]) or max(tumor_contours[i][:, 0, :][:, 1]) == min(tumor_contours[i][:, 0, :][:, 1]):
                    # pdb.set_trace()
                    print(tumor_contours[i][:, 0, :])
                    continue
                else:
                    dict_["name"] = "Annotation " + str(i)
                    dict_["points"] = (tumor_contours[i][:,0,:]*2**7).tolist()
                    json_array_copy["data"].append(dict_)

            json_arrra_save_path_one = tumor_wsi_json_base_path + "cut_region/" + os.path.basename(json_path)

            with io.open(json_arrra_save_path_one, 'w', encoding='utf8') as outfile:
                json_patch = json.dumps(json_array_copy)
                outfile.write(str(json_patch))
            mask_gen = MaskOnTheFly(json_arrra_save_path_one, yaml_file_path)

            for i in range(len(tumor_contours)):
                # pdb.set_trace()
                if len(tumor_contours[i][:, 0, :]) == 1:
                        # pdb.set_trace()
                    continue
                elif max(tumor_contours[i][:, 0, :][:, 0]) == min(tumor_contours[i][:, 0, :][:, 0]) or max(tumor_contours[i][:, 0, :][:, 1]) == min(tumor_contours[i][:, 0, :][:, 1]):
                    # pdb.set_trace()
                    print(tumor_contours[i][:, 0, :])
                    continue
                else:
                    tumor_hard_min_x = min(tumor_contours[i][:,0,:][:,0])
                    tumor_hard_min_y = min(tumor_contours[i][:,0,:][:,1])
                    tumor_hard_max_x = max(tumor_contours[i][:,0,:][:,0])
                    tumor_hard_max_y = max(tumor_contours[i][:,0,:][:,1])

                    tumor_hard_width = int((tumor_hard_max_x - tumor_hard_min_x))*2**7
                    tumor_hard_height = int(tumor_hard_max_y - tumor_hard_min_y)*2**7

                    tumor_hard_x_wsi = tumor_hard_min_x*2**7
                    tumor_hard_y_wsi = tumor_hard_min_y*2**7
                    tumor_hard_y_max = tumor_hard_max_y*2**7
                    tumor_hard_y_max = tumor_hard_max_x*2**7
                    x_y_list = []
                    if tumor_hard_width<224:
                        tumor_hard_width = 224
                    if tumor_hard_height<224:
                        tumor_hard_width = 224
                    for i in range(0,tumor_hard_width,224):
                        if i + 224 > tumor_hard_width and i > 224:
                            i = tumor_hard_width-224
                        else:
                            i = i
                        for j in range(0,tumor_hard_height,224):
                            if j +224 > tumor_hard_height and i>224:
                                j = tumor_hard_height -224
                                x_y_list.append((i ,j))
                            else:
                                x_y_list.append((i ,j))
                    x_y_list = list(set(x_y_list))
                    # pdb.set_trace()
                    # json_open = open(json_path)
                    # json_array = json.load(json_open)
                    # json_array["data"] = []

                    # mask_gen = MaskOnTheFly(json_path, yaml_file_path)
                    print("tumor_hard_width is {}".format(tumor_hard_width))
                    print("tumor_hard_height is {}".format(tumor_hard_height))
                    print((math.ceil(tumor_hard_width/224))*(math.ceil(tumor_hard_height/224)))
                    print(len(x_y_list))
                    if math.ceil(tumor_hard_width/224)*math.ceil(tumor_hard_height/224) != len(x_y_list):
                        sys.exit()
                    for x_y in x_y_list:
                        meta_dict = {"patch_id": "unknown",
                                    "img_path": "unknown",
                                    "mask_path": "unknown",
                                    "patch_size": [224,224],
                                    "fgmask_path": "unknown"
                                    } 
                        # img = slide.read_region(x_y,0,(224,224))
                        x_cut_id = int(x_y[0] + tumor_hard_x_wsi)
                        y_cut_id = int(x_y[1] + tumor_hard_y_wsi)
                        cut_patch_size_w = 224
                        cut_patch_size_h = 224
                        # print(x_cut_id)
                        # print(y_cut_id)
                        from PIL import Image
                        mask_np = mask_gen.draw_contours_on_the_fly((x_cut_id, y_cut_id), int(cut_patch_size_w), int(cut_patch_size_h), level=0, only_group_names=["tumor"])
                        mask_np = np.array(mask_np)
                        tumor_mask_nonzero_rate = np.count_nonzero(mask_np)/(cut_patch_size_w*cut_patch_size_h)
                        # print(tumor_mask_nonzero_rate)
                        # img_save_path = save_path + "/patch"
                        if tumor_mask_nonzero_rate > 0.7:
                            fgmask_path = os.path.join(fgmask_dir, "fgmask_" + meta_data_array["image_id"] + ".png")
                            fgmask_image = np.transpose(np.array(Image.open(fgmask_path)), (1,0))
                            fgmask_path_224 = extract_fg_patch(fgmask_image,y_cut_id,x_cut_id,cut_patch_size_w,cut_patch_size_h,level=0,level_fgmask=3)
                            # fgmask_path_224 = Image.fromarray(fgmask_path_224)
                            fg_img_nonzero_rate = np.count_nonzero(fgmask_path_224)/(cut_patch_size_w*cut_patch_size_h)
                            if fg_img_nonzero_rate > 0.5:
                                # pdb.set_trace()
                                img = slide.read_region((x_cut_id,y_cut_id),0,(224,224))
                                np_img = np.array(img)
                                img = np_img[:,:,0:3]
                                # fg_mask = extract_fg_patch()
                                meta_dict["patch_id"] = [x_cut_id,y_cut_id]

                                fgmask_path_224 = Image.fromarray(fgmask_path_224)
                                fgmask_save_path = save_path + "fgmask" + "/fgmask_" + os.path.basename(wsi_path) + "_level_0" + "_x_" + str(x_cut_id) + "_y_" + str(y_cut_id) +".png"
                                meta_dict["fgmask_path"] = fgmask_save_path
                                fgmask_path_224.save(fgmask_save_path)

                                tumor_mask = Image.fromarray(mask_np)
                                mask_save_path = save_path + "mask" + "/mask_" + os.path.basename(wsi_path) + "_level_0" + "_x_" + str(x_cut_id) + "_y_" + str(y_cut_id) +".png"
                                meta_dict["mask_path"] = mask_save_path
                                tumor_mask.save(mask_save_path)

                                img_pil = Image.fromarray(np_img[:,:,0:3])
                                img_save_path = save_path + "patch" + "/img_" + os.path.basename(wsi_path) + "_level_0" + "_x_" + str(x_cut_id) + "_y_" + str(y_cut_id) +".png"
                                meta_dict["img_path"] = img_save_path
                                img_pil.save(img_save_path)
                                meta_data_array["patches"].append(meta_dict)
            # pdb.set_trace()                    
            print(len(meta_data_array["patches"]))
        if label == "normal":
            patch_json_path = save_path + "json_fgmask"  + "/" + str(num) + "_json_" + os.path.basename(wsi_path) +  "_level_0" + '.json'
        else:
            patch_json_path = save_path + "json_fgmask" + "/" + str(num) +"_json_" + os.path.basename(wsi_path) +  "_level_0" + '.json'
        # save_json_path = os.path.join(json_base_path, wsi_name + "_level_0" + '.json')
        # if not os.path.exists(patch_json_path):
        #     os.makedirs(patch_json_path)

        if len(meta_data_array["patches"]) != 0 :
            with io.open(patch_json_path, 'w', encoding='utf8') as outfile:
                json_patch = json.dumps(meta_data_array)
                outfile.write(str(json_patch))
            slide.close()
            print('INFO: json save done!')
        else:
            print("pass")

