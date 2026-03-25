git clone https://github.com/Mechetel/Attention-Steganogan.git
cd Attention-Steganogan/
apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo scikit-image


cd data/

# =========================================================================
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

# =========================================================================
# Download MS-COCO 2017

mkdir mscoco
cd mscoco

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
# mkdir val
# mv test2017 val/_
# rm test2017.zip

# wget http://images.cocodataset.org/zips/train2017.zip
# unzip train2017.zip
# mkdir train
# mv train2017 train/_
# rm train2017.zip

mkdir -p val/_ train/_
ls test2017 | shuf -n 1000 | xargs -I{} mv test2017/{} val/_/
ls test2017 | shuf -n 1000 | xargs -I{} mv test2017/{} train/_/
rm test2017.zip


# =========================================================================
# Download CelebA

mkdir celeba
cd celeba
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
unzip celeba.zip
mkdir -p val/_ train/_
ls img_align_celeba | shuf -n 1000 | xargs -I{} mv img_align_celeba/{} val/_/
ls img_align_celeba | shuf -n 1000 | xargs -I{} mv img_align_celeba/{} train/_/
rm celeba.zip
