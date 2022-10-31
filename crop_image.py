from matplotlib.pyplot import box

try:
    from wanwu import Det, Backends
except ImportError:
    print(
        "wanwu API not opensourced yet, please stay tune! currently you can choose any other face detection model!"
    )
from alfred.utils.file_io import ImageSourceIter
import argparse
import cv2
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", help="onnx model")
    parser.add_argument("-i", "--img", default="demo/2.jpg", help="Test image file")
    parser.add_argument("-b", "--backend", default="onnxruntime", help="Runing backend")
    parser.add_argument("--show", action="store_true", help="vis result")
    args = parser.parse_args()

    det = Det(
        type="scrfd_500m",
        onnx_f=args.model_file,
        input_width=512,
        input_height=512,
        num_classes=1,
        score_thr=0.3,
        nms_thr=0.45,
        backend=Backends.from_str(args.backend),
        timing=True,
    )

    iter = ImageSourceIter(args.img)
    while True:
        raw_im = next(iter)
        if isinstance(raw_im, str):
            im = cv2.imread(raw_im)
        inp = det.load_inputs(im, normalize_255=False, is_rgb=True)
        boxes, scores, labels = det.infer(inp, im.shape[0], im.shape[1])

        # im_ = det.vis_res(im, boxes, scores, labels, class_names=["face"])

        for b in boxes:
            # expand
            b[0] = max(b[0] - (b[2] - b[0]) * 0.3, 0)
            b[1] = max(b[1] - (b[3] - b[1]) * 0.3, 0)
            b[2] = b[2] + (b[2] - b[0]) * 0.3
            b[3] = b[3] + (b[3] - b[1]) * 0.3
            b = np.array(b).astype(np.int)
            target_im = np.zeros([256, 256, 3]).astype(np.uint8)
            print(b)
            cropped_im = im[b[1] : b[3], b[0] : b[2], :]
            r = 256 / max(cropped_im.shape)
            cropped_im_resized = cv2.resize(cropped_im, dsize=None, fx=r, fy=r)
            print(cropped_im_resized.shape)

            l_y = target_im.shape[0] - cropped_im_resized.shape[0]
            l_x = int((target_im.shape[1] - cropped_im_resized.shape[1]) / 2)
            print(l_y, l_x)
            target_im[
                l_y:, l_x : cropped_im_resized.shape[1] + l_x
            ] = cropped_im_resized

            cv2.imshow("original", im)
            cv2.imshow("cropped_im", cropped_im)
            cv2.imshow("cropped_im_resized", cropped_im_resized)
            cv2.imshow("target_im", target_im)

            save_p = os.path.join(
                "data",
                os.path.basename(raw_im).split(".")[0] + f"_{iter.crt_index}.png",
            )
            cv2.imwrite(save_p, target_im)

        cv2.waitKey(0)
