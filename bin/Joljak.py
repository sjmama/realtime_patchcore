import contextlib
import gc
import logging
import os
import sys
import time
import click
import numpy as np
import torch
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
import cv2 as cv
import threading
LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}
cam=cv.VideoCapture("rtmp://ip/test_live/0000")
status, img = cam.read()
height, width = img.shape[:2]
ratio = 320 / height
resized_image = cv.resize(img, (int(width * ratio), 320))
img = resized_image[:, (resized_image.shape[1]-320)//2:(resized_image.shape[1]+320)//2]
img = np.transpose(img, (2,0,1))
img=img/255.0
img=torch.tensor(img).to(torch.float)
img=img.unsqueeze(0)
@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)

def main(**kwargs):
    pass
flag=False
@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: item for (key, item) in methods}
    os.makedirs(results_path, exist_ok=True)
    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    
    # dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    # dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    #여기에 캡쳐 반복문 넣으면 될듯
    nums=1
    PatchCore_list = next(patchcore_iter)
    while cam.isOpened():
        while flag!=True:
            print(flag)
            time.sleep(0.1)
        time.sleep(2.0)
        # status, img = cam.read()###
        # if not status:
        #     time.sleep(0.1)
        #     continue
        # height, width = img.shape[:2]
        # ratio = 320 / height
        # resized_image = cv.resize(img, (int(width * ratio), 320))
        # img = resized_image[:, (resized_image.shape[1]-320)//2:(resized_image.shape[1]+320)//2]
        # img = np.transpose(img, (2,0,1))
        # img=img/255.0
        # img=torch.tensor(img).to(torch.float)
        # img=img.unsqueeze(0)
        
        # for dataloader_count, dataloaders in enumerate(dataloader_iter):
        #     LOGGER.info(
        #         "Evaluating dataset [{}] ({}/{})...".format(
        #             dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
        #         )
        #     )
        
        
        patchcore.utils.fix_seeds(seed, device)

        #dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                
                
                scores, segmentations  = PatchCore.predict(###################여기서 프레딕트가 뭐하는건지 알아야됨 여기다 넣어주는거다 데이터로더면 데이터로더로 아니면 이미지로처리
                    img###
                    #dataloaders["testing"]##여기에 이미지 스트림에서 캡쳐한거 넣으면 될듯
                )##labels_gt, masks_gt 이거 두개도  predic로 데이터로더 들어가면 나오는건데 이거 관련된 변수들 다 지움
                print("score : ",scores)
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            scores = np.array(aggregator["scores"])
            min_scores = scores.min()
            max_scores = scores.max()
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)
            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # anomaly_labels = [
            #     x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            # ]

            # Plot Example Images.
            if save_segmentation_images:
                # image_paths = [
                #     x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                # ]
                # mask_paths = [
                #     x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                # ]

                # def image_transform(image):
                #     in_std = np.array(
                #         dataloaders["testing"].dataset.transform_std
                #     ).reshape(-1, 1, 1)
                #     in_mean = np.array(
                #         dataloaders["testing"].dataset.transform_mean
                #     ).reshape(-1, 1, 1)
                #     image = dataloaders["testing"].dataset.transform_img(image)
                #     return np.clip(
                #         (image.numpy() * in_std + in_mean) * 255, 0, 255
                #     ).astype(np.uint8)

                # def mask_transform(mask):
                #     return dataloaders["testing"].dataset.transform_mask(mask).numpy()
                
                patchcore.utils.plot_segmentation_images(
                    results_path,
                    img,
                    segmentations,
                    scores,
                    num=nums
                )
                nums+=1
                if nums>10000:
                    nums=1
            LOGGER.info("Computing evaluation metrics.")
            # Compute Image-level AUROC scores for all images.

            # Compute PRO score & PW Auroc for all images

            # Compute PRO score & PW Auroc only for images with anomalies
            sel_idxs = []
            
            anomaly_pixel_auroc = 0
            full_pixel_auroc=0
            auroc=0
            print(f"num : {nums} mean score : {scores}")
            result_collect.append(
                {
                    "dataset_name": "test",
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            #del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])

def cap():
    global flag
    global img
    while True:
        status, iimg = cam.read()###
        flag=True
        if not status:
            time.sleep(0.1)
            continue
        height, width = iimg.shape[:2]
        ratio = 320 / height
        resized_image = cv.resize(iimg, (int(width * ratio), 320))
        iimg = resized_image[:, (resized_image.shape[1]-320)//2:(resized_image.shape[1]+320)//2]
        iimg = np.transpose(iimg, (2,0,1))
        iimg=iimg/255.0
        iimg=torch.tensor(iimg).to(torch.float)
        img=iimg.unsqueeze(0)
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    capt = threading.Thread(target=cap)
    capt.start()
    main()
