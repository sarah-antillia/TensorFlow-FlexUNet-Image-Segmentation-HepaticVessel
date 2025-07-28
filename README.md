<h2>TensorFlow-FlexUNet-Image-Segmentation-HepaticVessel (2025/07/29)</h2>

This is the first preliminary experiment of Image Segmentation for HepaticVessel Multiclass (Vessel and Tumor) based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1r2f3wB13QY-1KMvkU-fZbYDFGKx_AHvc/view?usp=sharing">
HepaticVessel-ImageMask-Dataset.zip</a>.
which was derived by us from <br>
<a href="https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=drive_link">
Task08_HepaticVessel.tar
</a>
on google drive <a href="https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2">
Medical Segmentation Decathlon (MSD)
</a>


<br>
<br>
<b>Acutual Image Segmentation for 512x512 HepaticVessel images</b><br>
As shown below, the inferred masks look very similar to the ground truth masks. 
The green region represents a vessel, and the red a tumor respectively.<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11485_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11485_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11485_16.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11596_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11596_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11596_19.png" width="320" height="auto"></td>

</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11770_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11770_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11770_27.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the following dataset:<br>
<a href="https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=drive_link">
Task08_HepaticVessel.tar
</a>
on the google drive <a href="https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2">
Medical Segmentation Decathlon (MSD)
</a>
<br>

<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by-sa/4.0/deed.en"
<b>CC-BY-SA 4.0</b>
</a>
<br>

All data will be made available online with a permissive copyright-license (CC-BY-SA 4.0), <br>
allowing for data to be shared, distributed and improved upon. <br>
All data has been labeled and verified by an expert human rater, and with the best effort to <br>
mimic the accuracy required for clinical use. <br>
To cite this data, please refer to 
<a href="https://arxiv.org/abs/1902.09063">https://arxiv.org/abs/1902.09063
</a>
<br>
<h3>
<a id="2">
2 HepaticVessel ImageMask Dataset
</a>
</h3>
 If you would like to train this HepaticVessel Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1r2f3wB13QY-1KMvkU-fZbYDFGKx_AHvc/view?usp=sharing">
HepaticVessel-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─HepaticVessel
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>HepaticVessel Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/HepaticVessel/HepaticVessel_Statistics.png" width="512" height="auto"><br>
<br>
On the derivation of the dataset, please refer to our repository 
<a href="https://github.com/sarah-antillia/Image-Segmentation-Liver-Tumor">Image-Segmentation-Liver-Tumor</a>
<br><br>
<!--
On the derivation of the dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
-->
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained HepaticVessel TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/HepaticVessel/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/HepaticVessel and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for HepaticVessel 1+2 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; background   black  vessel:green, tumor:red
rgb_map = {(0,0,0):0, (0,255,0):1, (255,0,0):2,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 7,8,9)</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 18.<br><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/train_console_output_at_epoch18.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/HepaticVessel/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/HepaticVessel/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/HepaticVessel</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for HepaticVessel.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/evaluate_console_output_at_epoch18.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/HepaticVessel/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this HepaticVessel/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0057
dice_coef_multiclass,0.997
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/HepaticVessel</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for HepaticVessel.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/HepaticVessel/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11485_29.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11485_29.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11485_29.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11653_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11653_30.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11653_30.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11711_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11711_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11711_12.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/11711_39.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/11711_39.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/11711_39.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/12016_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/12016_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/12016_16.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/images/12080_26.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test/masks/12080_26.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HepaticVessel/mini_test_output/12080_26.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1.Medical Segmentation Decathlon Generalisable 3D Semantic Segmentation</b><br>
<a href="http://medicaldecathlon.com/">http://medicaldecathlon.com/</a>
<br>
<br>
<b>2. MSD Hepatic Vessel</b><br>
OpenMEDLab<br>
<a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MSD_Hepatic_Vessel.md">
MSD Hepatic Vessel
</a>
<br>
<br>
<b>3. A large annotated medical image dataset for the<br>
development and evaluation of segmentation algorithms</b><br>
Amber L. Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani, Bram van Ginneken, <br>
Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers,<br>
 Patrick Bilic, Patrick F. Christ, Richard K. G. Do, Marc Gollub, Jennifer Golia-Pernicka, Stephan H. Heckers,<br>
  William R. Jarnagin, Maureen K. McHugo, Sandy Napel, Eugene Vorontsov, Lena Maier-Hein, M. Jorge Cardoso<br>
<a href="https://arxiv.org/pdf/1902.09063">https://arxiv.org/pdf/1902.09063</a>

<br>
<br>
<b>4. Hepatic vessels segmentation using deep learning and preprocessing enhancement</b><br>
Omar Ibrahim Alirr, Ashrani Aizzuddin Abd Rahni<br>
<a href="https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13966">https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13966</a>
