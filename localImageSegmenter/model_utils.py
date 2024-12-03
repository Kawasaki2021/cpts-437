import os
import time
import glob
import numpy as np
import torch
from torch.nn.functional import softmax
from skimage.io import imread
from skimage import img_as_float32
import im_utils
from rootPainterModels import UNetGNRes
from metrics import get_metrics
from file_utils import ls
from loss import combined_loss as criterion


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device('cuda')


device = get_device() # used in epoch function etc.

def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths

def load_model(model_path):
    model = UNetGNRes()
    if torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(model_path))
            model = torch.nn.DataParallel(model)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        # if you are running on a CPU-only machine, please use torch.load with 
        # map_location=torch.device('cpu') to map your storages to the CPU.
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model = torch.nn.DataParallel(model)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def create_first_model_with_random_weights(model_dir):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes()
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)

    if torch.cuda.is_available():
        model.cuda()
    return model


def get_prev_model(model_dir):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path)
    return prev_model, prev_path

def get_val_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, bs):
    """
    Return the TP, FP, TN, FN, defined_sum, duration
    for the {cnn} on the validation set

    TODO - This is too similar to the train loop. Merge both and use flags.
    """
    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    # TODO: In order to speed things up, be a bit smarter here
    # by only segmenting the parts of the image where we have
    # some annotation defined.
    # implement a 'partial segment' which exlcudes tiles with no
    # annotation defined.
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_sum = 0
    for fname in fnames:
        annot_path = os.path.join(val_annot_dir,
                                  os.path.splitext(fname)[0] + '.png')
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print(f'Exception reading annotation {annot_path} inside validation method.'
                  'Will retry in 0.1 seconds')
            print(fname, ex)
            time.sleep(0.1)
            annot = imread(annot_path)

        annot = np.array(annot)
        foreground = annot[:, :, 0].astype(bool).astype(int)
        background = annot[:, :, 1].astype(bool).astype(int)
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])

        # Use glob.escape to allow arbitrary strings in file paths,
        # including [ and ]  
        # For related bug See https://github.com/Abe404/root_painter/issues/87
        image_path_part = glob.escape(image_path_part)

        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        image, pad_settings = im_utils.pad_to_min(image, min_w=572, min_h=572)
        predicted = unet_segment(cnn, image, bs, in_w,
                                 out_w, threshold=0.5)
        predicted = im_utils.crop_from_pad_settings(predicted, pad_settings)

        # mask defines which pixels are defined in the annotation.
        mask = foreground + background
        mask = mask.astype(bool).astype(int)
        predicted *= mask
        predicted = predicted.astype(bool).astype(int)
        y_defined = mask.reshape(-1)
        y_pred = predicted.reshape(-1)[y_defined > 0]
        y_true = foreground.reshape(-1)[y_defined > 0]
        tps += np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tns += np.sum(np.logical_and(y_pred == 0, y_true == 0))
        fps += np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fns += np.sum(np.logical_and(y_pred == 0, y_true == 1))
        defined_sum += np.sum(y_defined > 0)
    duration = round(time.time() - start, 3)
    metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
    return metrics

def save_if_better(model_dir, cur_model, prev_model_path,
                   cur_f1, prev_f1):
    print('prev f1', str(round(prev_f1, 5)).ljust(7, '0'),
          'cur f1', str(round(cur_f1, 5)).ljust(7, '0'))
    if cur_f1 > prev_f1:
        prev_model_fname = os.path.basename(prev_model_path)
        prev_model_num = int(prev_model_fname.split('_')[0])
        model_num = prev_model_num + 1
        now = int(round(time.time()))
        model_name = str(model_num).zfill(6) + '_' + str(now) + '.pkl'
        model_path = os.path.join(model_dir, model_name)
        print('saving', model_path, time.strftime('%H:%M:%S', time.localtime(now)))
        torch.save(cur_model.state_dict(), model_path)
        return True
    return False

def ensemble_segment(model_paths, image, bs, in_w, out_w,
                     threshold=0.5):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    image, pad_settings = im_utils.pad_to_min(image, min_w=in_w, min_h=in_w)
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = load_model(model_path)
        preds = unet_segment(cnn, image,
                             bs, in_w, out_w, threshold=None)
        if pred_sum is not None:
            pred_sum += preds
        else:
            pred_sum = preds
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred = unet_segment(cnn, flipped_im, bs, in_w,
                                    out_w, threshold=None)
        pred_sum += np.fliplr(flipped_pred)
        pred_count += 1
    pred_sum = im_utils.crop_from_pad_settings(pred_sum, pad_settings)
    foreground_probs = pred_sum / pred_count
    predicted = foreground_probs > threshold
    predicted = predicted.astype(int)
    return predicted


def epoch(model, train_loader, batch_size,
          optimizer, step_callback, stop_fn):
    """ One training epoch """
    
    model.to(device)
    model.train()
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_total = 0

    for step, (photo_tiles,
               foreground_tiles,
               defined_tiles) in enumerate(train_loader):

        photo_tiles = photo_tiles.to(device)
        foreground_tiles = foreground_tiles.to(device).float()
        defined_tiles = defined_tiles.to(device)
        optimizer.zero_grad()

        outputs = model(photo_tiles)
        softmaxed = softmax(outputs, 1)

        # just the foreground probability. (remove soon)
        foreground_probs = softmaxed[:, 1, :]

        outputs[:, 0] *= defined_tiles
        outputs[:, 1] *= defined_tiles

        loss = criterion(outputs, foreground_tiles.long())

        loss.backward()
        optimizer.step()

        if step_callback:
            step_callback()

        # This bit assumes we do softmax on the model output (not yet implemented)
        # make the predictions match in undefined areas so metrics in these
        # regions are not taken into account.
        #outputs *= defined_tiles 
        #predicted = outputs > 0.5

        # we only want to calculate metrics on the
        # part of the predictions for which annotations are defined
        # so remove all predictions and foreground labels where
        # we didn't have any annotation.

        defined_list = defined_tiles.reshape(-1)
        preds_list = foreground_probs.reshape(-1)[defined_list > 0] > 0.5
        foregrounds_list = foreground_tiles.reshape(-1)[defined_list > 0]

        # # calculate all the false positives, false negatives etc
        tps += torch.sum((foregrounds_list == 1) * (preds_list == 1)).cpu().numpy()
        tns += torch.sum((foregrounds_list == 0) * (preds_list == 0)).cpu().numpy()
        fps += torch.sum((foregrounds_list == 0) * (preds_list == 1)).cpu().numpy()
        fns += torch.sum((foregrounds_list == 1) * (preds_list == 0)).cpu().numpy()
        defined_total += torch.sum(defined_list > 0).cpu().numpy()
        # https://github.com/googlecolab/colabtools/issues/166
        print(f"\rTraining: {(step+1) * batch_size}/"
                f"{len(train_loader.dataset)} "
                f" loss={round(loss.item(), 3)}",
                end='', flush=True)
        if stop_fn and stop_fn():
            return None
    return (tps, fps, tns, fns, defined_total)

def unet_segment(cnn, image, bs, in_w, out_w, threshold=0.5):
    """
    Threshold set to None means probabilities returned without thresholding.
    """
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    tiles, coords = im_utils.get_tiles(image,
                                       in_tile_shape=(in_w, in_w, 3),
                                       out_tile_shape=(out_w, out_w))
    tile_idx = 0
    batches = []
    while tile_idx < len(tiles):
        tiles_to_process = []
        for _ in range(bs):
            if tile_idx < len(tiles):
                tile = tiles[tile_idx]
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                tile_idx += 1
                tiles_to_process.append(tile)
        tiles_for_gpu = torch.from_numpy(np.array(tiles_to_process))
        if torch.cuda.is_available():
            tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.float()
        batches.append(tiles_for_gpu)

    output_tiles = []
    for gpu_tiles in batches:
        
        if(torch.cuda.is_available()):

            outputs = cnn(gpu_tiles.cuda())

        else:

            outputs = cnn(gpu_tiles.cpu())

        softmaxed = softmax(outputs, 1)
        foreground_probs = softmaxed[:, 1, :]  # just the foreground probability.
        if threshold is not None:
            predicted = foreground_probs > threshold
            predicted = predicted.view(-1).int()
        else:
            predicted = foreground_probs

        pred_np = predicted.data.cpu().numpy()
        out_tiles = pred_np.reshape((len(gpu_tiles), out_w, out_w))
        for out_tile in out_tiles:
            output_tiles.append(out_tile)

    assert len(output_tiles) == len(coords), (
        f'{len(output_tiles)} {len(coords)}')

    reconstructed = im_utils.reconstruct_from_tiles(output_tiles, coords,
                                                    image.shape[:-1])
    return reconstructed