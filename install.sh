git clone https://github.com/Mechetel/Attention-Steganogan.git
cd Attention-Steganogan/
apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo scikit-image


cd data/

# Download DIV2K

mkdir div2k
cd div2k

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
mkdir val
unzip -j DIV2K_valid_HR.zip -d val/_
rm DIV2K_valid_HR.zip

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
mkdir train
unzip -j DIV2K_train_HR.zip -d train/_
rm DIV2K_train_HR.zip

# Download MS-COCO 2017

mkdir mscoco
cd mscoco

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mkdir val
mv test2017 val/_
rm test2017.zip

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mkdir train
mv train2017 train/_
rm train2017.zip


# if i will only make wget http://images.cocodataset.org/zips/test2017.zip and unzip test2017.zip in /workspace/Attention-Steganogan/data/mscoco/
# i want to make val/_ and train/_ folders in /workspace/Attention-Steganogan/data/mscoco/
# and move 1000 random images from test2017 to val/_ and 1000 images from train2017 to train/_
# create sh script to do this
cd /workspace/Attention-Steganogan/data/mscoco/
mkdir val
mkdir train
for i in {1..1000}
do
  mv test2017/$(ls test2017 | shuf -n 1) val/_
  mv train2017/$(ls train2017 | shuf -n 1) train/_
done
