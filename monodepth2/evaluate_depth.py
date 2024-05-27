from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import networks
import matplotlib.pyplot as plt
from kitti_dataset import RealSenseDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

#TODO: CORREGIR RUTA A MAPAS DE PROFUNDIDAD VERDADEROS


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

# Función para visualizar las profundidades
def visualize_depths(gt_depth, pred_depth, index):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'Ground Truth Depth - Index {index}')
    plt.imshow(gt_depth, cmap='plasma_r')
    plt.colorbar(label='Depth (m)')
    plt.subplot(1, 2, 2)
    plt.title(f'Predicted Depth - Index {index}')
    plt.imshow(pred_depth, cmap='plasma_r')
    plt.colorbar(label='Depth (m)')
    plt.show()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None: # si no existen predicciones almacenadas con anterioridad, calcularlas de nuevo

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files2.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = RealSenseDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))
               
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                
                pred_disps.append(pred_disp)
    

        pred_disps = np.concatenate(pred_disps, axis=0)
        

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]


    if opt.save_pred_disps:
        output_path = os.path.join(opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)


    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    #calculo de errores entre profundidades predichas por el modelo
    # y ground truth
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp #profundidad inversamente proporcional a la disparidad 

        if len(gt_depth.shape) == 3:
            gt_depth = gt_depth[:, :, 0]
        if len(pred_depth.shape) == 3:
            pred_depth = pred_depth[:, :, 0]

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Comprobación de arrays vacíos
        if pred_depth.size == 0 or gt_depth.size == 0:
            print(f"Empty depth arrays at index {i}")
            continue

        print(f"Index {i}:")
        print(f"  gt_depth.shape: {gt_depth.shape}")
        print(f"  pred_depth.shape: {pred_depth.shape}")
        print(f"  gt_depth min: {gt_depth.min()}, max: {gt_depth.max()}")
        print(f"  pred_depth min: {pred_depth.min()}, max: {pred_depth.max()}")


        pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            print(f"  Scaling ratio at index {i}: {ratio}")
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    # Visualizar algunas imágenes
    for i in range(25):  # Visualizar las primeras 5 imágenes
        gt_depth = gt_depths[i]
        pred_disp = pred_disps[i]
        pred_depth = 1 / pred_disp
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))
        visualize_depths(gt_depth, pred_depth, i)


#TODO: COMPROBAR QUE SE LE ESTEN PASANDO A LA RED LAS IMAGENES ORDENADAS, NO SE PUEDEN SORTEAR CREO
#TODO: COMPROBAR LAS ESCALAS, DEBEN COINCIDIR

if __name__ == "__main__":
    options = MonodepthOptions()

    opts = options.parse()

    opts.eval_mono = True  # o eval_stereo = True, dependiendo de tu caso
    opts.eval_stereo = False  # Asegúrate de que sólo uno sea True
    opts.data_path = "/home/andrea/Escritorio/TFG/monodepth2/datasets/"
    opts.load_weights_folder = "/home/andrea/Escritorio/TFG/monodepth2/models/mono+stereo_640x192" 

    # Verifica que la carpeta de pesos exista
    assert os.path.isdir(opts.load_weights_folder), "Cannot find folder at {}".format(opts.load_weights_folder)

    evaluate(opts)

    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())



