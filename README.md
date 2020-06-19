# English-Vietnamese NMT Demo
Github chứa project DATN  

## 1. Cấu trúc thư mục
- core2core : khai báo vấn đề dịch lõi Anh - Việt  
- corebooster : khai báo vấn đề tăng cường câu  
- data : dữ liệu sử dụng cho quá trình huấn luyện và đánh gía 
    + core2core : 
        * core_v1.tar.gz : dùng cho train  
        * core_v1_test.tar.gz : dùng cho test  
    + corebooster:  
        * corebooster_train.tar.gz : dùng cho train   
        * corebooster_test.tar.gz : dùng cho test  
    + EVBNews_2_0_transformer_200k.csv : dữ liệu tổng hợp    
 
## 2. Hướng dẫn
- Huấn luyện và đánh gía vấn đề dịch Anh-Việt [Translate En-Vi](https://colab.research.google.com/drive/1wnkrBRxT7_rvvhwW6H1tsfcKfcP-cE_1)  
- Huấn luyện và đánh gía vấn để dịch câu lõi [Core2core](https://colab.research.google.com/drive/1wcZx9XhykXJp9hL0EBGk1PUgYS_k_V4v)  
- Huấn luyện và đánh gía vấn đề tăng cường câu [CoreBoosting]()  
- Demo tải và sử dụng mô hình bằng Python [Colab Demo](https://colab.research.google.com/drive/1M_wTjTCpiqaiIhUt_eDh-0lDr1-Mkntf#scrollTo=2UUlItsTOVjn)

## 3. Tham khảo  
Ý tưởng của project dựa trên vấn đề dịch Anh-Việt TranslateEnviIWSLT32 tại [link](https://github.com/stefan-it/nmt-en-vi).  
Huấn luyện, đánh gía và sử dụng mô hình dựa trên framework [Tensor2tensor](https://tensorflow.github.io/tensor2tensor/)