# myssr-net
<li>References:</li>
   &nbsp;&nbsp;&nbsp;SSR-Net Github Repository:https://github.com/shamangary/SSR-Net<br>
<li>Data pre-processing：训练时需要将图片设置为（92*92），input中的csv文件可以直接作为输入文件使用,同时在input/make_dataset.py是相应制作数据集的方式</li>
   &nbsp;&nbsp;&nbsp;Download mdb dataset from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/. and unzip them under './input/imdb'<br>
   &nbsp;&nbsp;&nbsp;Download wiki dataset from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/. and unzip them under './input/wiki'<br>
   &nbsp;&nbsp;&nbsp;Download MegaAge-Asian dataset from http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/ and unzip them under './input/asian'<br>
<li>Dependencies:</li>
   &nbsp;&nbsp;&nbsp;opencv<br>
   &nbsp;&nbsp;&nbsp;pytorch<br>
<li>Training:</li>
&nbsp;&nbsp;&nbsp;python3 ssr_train.py&nbsp;&nbsp;&nbsp;and the min MAELoss is 4.32<br>
video.py使用摄像头进行验证


