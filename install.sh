git clone https://github.com/Mechetel/Attention-Steganogan.git
cd Attention-Steganogan/
apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo scikit-image


cd data/
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
