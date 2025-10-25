# <p dir="rtl" align="justify">1. سیستم تشخیص کلاهبرداری کارت اعتباری با الگوریتم Naive Bayes</p>

### <p dir="rtl" align="justify">معرفی پروژه:</p>
<p dir="rtl" align="justify">این پروژه یک سیستم تشخیص کلاهبرداری کارت اعتباری با استفاده از الگوریتم Naive Bayes پیاده‌سازی کرده است. هدف اصلی شناسایی تراکنش‌های متقلبانه با دقت بالا است.</p>

### <p dir="rtl" align="justify">اهداف پروژه:</p>
<ul dir="rtl" align="justify">
   <li> ساخت مدل یادگیری ماشین برای تشخیص کلاهبرداری</li>
   <li> مدیریت عدم تعادل کلاس‌ها در دیتاست</li>
   <li> ارزیابی عملکرد مدل با معیارهای مختلف</li>
   <li> ارائه راهکارهای بهبود مدل</li>
</ul>

### <p dir="rtl" align="justify">مشخصات دیتاست:</p>
<ul dir="rtl" align="justify">
   <li>منبع: Kaggle Credit Card Fraud Detection</li>
   <li>تعداد نمونه: 284,807 تراکنش</li>
   <li>تعداد ویژگی‌ها: 31 ویژگی</li>
   <li>تراکنش‌های عادی: 284,315 (99.83%)</li>
   <li>تراکنش‌های متقلبانه: 492 (0.17%)</li>
</ul>

### <p dir="rtl" align="justify">ویژگی‌های دیتاست:</p>
<ul dir="rtl" align="justify">
   <li>ویژگی‌های V1 تا V28: نتایج تحلیل PCA (محافظت‌شده)</li>
   <li>Time: زمان تراکنش</li>
   <li>Amount: مبلغ تراکنش</li>
   <li>Class: برچسب (0=عادی, 1=متقلبانه)</li>
</ul>


### <p dir="rtl" align="justify">توزیع کلاس‌ها:</p>
<p dir="rtl" align="justify">دیتاست دارای عدم تعادل شدید است:</p>
<ul dir="rtl" align="justify">
   <li>تراکنش‌های عادی: 99.83%</li>
   <li>تراکنش‌های متقلبانه: 0.17%</li>
</ul>

### <p dir="rtl" align="justify">مراحل پیش‌پردازش:</p>
<ul dir="rtl" align="justify">
   <li>حذف داده‌های تکراری: 1,081 نمونه تکراری حذف شد</li>
   <li>نرمال‌سازی: استانداردسازی ویژگی‌های Time و Amount با RobustScaler</li>
   <li>مدیریت عدم تعادل: استفاده از RandomUnderSampler</li>
</ul>

### <p dir="rtl" align="justify">نمونه‌گیری مجدد</p>
<ul dir="rtl" align="justify">
   <li>قبل از نمونه‌گیری: 283,253 عادی و 473 متقلبانه</li>
   <li>بعد از نمونه‌گیری: 946 عادی و 473 متقلبانه</li>
</ul>

### <p dir="rtl" align="justify">الگوریتم مورد استفاده:</p>
<ul dir="rtl" align="justify">
   <li>الگوریتم: Gaussian Naive Bayes</li>
   <li>تقسیم داده: 70% آموزش، 30% آزمون</li>
   <li>استانداردسازی: StandardScaler</li>
</ul>

### <p dir="rtl" align="justify">نتایج روی داده متعادل‌شده:</p>

| Metric | Value |
|--------|-------|
| Accuracy | 92.49% |
| Recall | 81.69% |
| Precision | 95.08% |
| F1-Score | 87.88% |

# 2. run project:
### 1. Create a virtual environment in project root not src/ (windows):
```
python -m venv venv
.\venv\Scripts\activate
```

### or
### Create a virtual environment in project root not src/ (linux)
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install requirements:
```
pip install -r requirements.txt
```

### 3. 
```
cd src
```
### windows:
```
python .\credit-card-fraud-detection.py
```
