### 환경 설정
python = 3.7.13
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
추가 패키지 설치: scipy, tqdm, scikit-learn

### 데이터 전처리
mat 파일 기준 prepare_mat.py으로 처리 가능
채널은 지금 고정 format으로 정해져 있음. (eeg_data, label) 

입력 예시:
cd prepare_dataset
python prepare_mat --data_dir "" --output_dir ""


### Train
train_Kfold_CV

