# Nuray-Huseynova-
# CTRL IS HERS - Data Science Project

## Layihə haqqında

Bu layihəni Kaggle-da keçirilən CTRL IS HERS Final yarışması üçün hazırlamışam. İşə başlayanda məqsədim datasetin necə qurulduğunu anlamaq, məlumatları analiz etmək və `target` dəyişənini proqnozlaşdırmaq üçün müxtəlif Machine Learning modellərini sınamaq idi.

İlk olaraq train və test fayllarını yoxladım. Daha sonra məlumatların təhlili ilə məşğul oldum, tarix sütunlarından yeni xüsusiyyətlər yaratdım və bir neçə model quraraq nəticələrini müqayisə etdim. Son mərhələdə isə modellərin proqnozlarını birləşdirməyin nəticəyə təsirini yoxladım.

Layihə üzərində Python və Google Colab vasitəsilə işləmişəm.

## Dataset

Kaggle yarışmasının təqdim etdiyi train və test datasetlərindən istifadə etmişəm.

| Göstərici | Dəyər |
|---|---:|
| Train datası | 31 715 sətir, 49 sütun |
| Test datası | 7 929 sətir, 48 sütun |
| `target = 0` | 14 358 (45,27%) |
| `target = 1` | 17 357 (54,73%) |
| Çatışmayan dəyərlər | Tapılmayıb |
| Təkrarlanan sətirlər | Tapılmayıb |
| Train məlumatlarının tarix aralığı | 07.01.2013 - 28.08.2014 |
| Test məlumatlarının tarix aralığı | 28.08.2014 - 27.12.2014 |

`target` sütununda iki qiymət var: 0 və 1. Train datasında `target = 0` olan 14 358, `target = 1` olan isə 17 357 sətir mövcuddur. Yəni siniflərin sayı bir-birinə yaxındır, amma tam bərabər deyil.

Datasetdə tarix, həftənin günü, kanal və müxtəlif rəqəmsal göstəricilər kimi xüsusiyyətlər var. Train datasında `target` sütunu da olduğu üçün onun sütun sayı test datasından bir vahid çoxdur.

## Məlumatların ilkin yoxlanılması və analizi

İlk mərhələdə datasetin ölçüsünə, sütunlarına və məlumat tiplərinə baxdım. Çatışmayan dəyərləri və təkrarlanan sətirləri də yoxladım. Apardığım yoxlamalarda bunlara rast gəlmədim.

Sonra `target` dəyişəninin paylanmasını və digər xüsusiyyətləri araşdırdım. Məlumatların daha yaxşı başa düşülməsi üçün kateqorik və rəqəmsal sütunlara ayrıca baxdım.

Tarix məlumatlarından aşağıdakı yeni sütunları yaratdım:

- `publish_year`
- `publish_month`
- `publish_day`
- `publish_dayofyear`

Bundan başqa, `weekday` və `channel` kimi kateqorik xüsusiyyətlərin modellərdə düzgün istifadə olunmasına diqqət yetirdim.

Rəqəmsal sütunları yoxlayanda bəzi dəyişənlərdə qiymətlərin çox qeyri-bərabər paylandığını gördüm. Buna `kw_max_min`, `kw_avg_min` və `self_reference_min_shares` sütunlarını misal göstərmək olar.

EDA zamanı `target` paylanmasını, tarixlə bağlı xüsusiyyətləri, rəqəmsal dəyişənləri və model üçün faydalı ola biləcək göstəriciləri araşdırdım.

## Feature Engineering

Modelin nəticəsini yaxşılaşdırmaq üçün əlavə xüsusiyyətlər yaratmağı da sınadım. Müxtəlif nisbət göstəriciləri, sentiment göstəriciləri arasındakı fərqlər və tarix üçün dövri xüsusiyyətlər üzərində işlədim.

Ancaq bu dəyişikliklər gözlədiyim nəticəni vermədi. Həmin variantda ROC-AUC 0.74926, Log Loss isə 0.60495 oldu. Əvvəlki xüsusiyyət dəsti ilə nəticələr daha yaxşı olduğu üçün əlavə etdiyim xüsusiyyətləri yekun modeldə saxlamadım.

Bu mərhələdə nəticəni yaxşılaşdırmaq üçün etdiyim hər dəyişikliyin həqiqətən faydalı olub-olmadığını ayrıca yoxladım. Sonda daha yaxşı nəticə verən əvvəlki xüsusiyyət dəsti ilə davam etdim.

## Train və validation bölgüsü

Məlumatları bölərkən tarix ardıcıllığını nəzərə aldım. Təsadüfi bölgü etmək əvəzinə əvvəlki tarixlərə aid məlumatlarla modeli öyrədib, daha sonrakı tarixlərə aid məlumatlarla yoxladım.

Bu bölgüdə təxminən 24 479 sətir train, 7 236 sətir isə validation üçün istifadə olundu.

Bu üsulu seçməyimin səbəbi modelin gələcək tarixlərə aid məlumatlarda necə nəticə göstərdiyini yoxlamaq idi. Beləliklə, modeli yalnız öyrəndiyi məlumatlar üzərində deyil, ayrıca saxladığım validation hissəsində də qiymətləndirdim.

## Sınadığım Machine Learning modelləri

İlk olaraq Logistic Regression modeli ilə başladım. Daha sonra Random Forest, Gradient Boosting, CatBoost və LightGBM modellərini sınadım. XGBoost-u da ayrıca yoxladım və sonradan onun proqnozlarından digər modellərlə birlikdə istifadə etdim.

Modellərin nəticələrini eyni validation datası üzərində müqayisə etdim.

| Model | ROC-AUC | Log Loss |
|---|---:|---:|
| Logistic Regression | 0.7291 | 0.6178 |
| Random Forest | 0.7360 | 0.6094 |
| Gradient Boosting | 0.7432 | 0.6051 |
| CatBoost | 0.7497 | 0.6005 |
| LightGBM | 0.7477 | 0.6001 |

Logistic Regression modeli üçün digər nəticələr də belə oldu:

| Metrika | Nəticə |
|---|---:|
| Accuracy | 0.6797 |
| Precision | 0.7197 |
| Recall | 0.6767 |
| F1-score | 0.6975 |
| ROC-AUC | 0.7291 |
| Log Loss | 0.6178 |

Nəticələri müqayisə edəndə CatBoost-un fərdi modellər arasında ən yüksək ROC-AUC verdiyini gördüm. LightGBM-in Log Loss göstəricisi isə CatBoost-dan bir qədər aşağı oldu.

Burada iki metrikaya diqqət yetirdim. ROC-AUC modelin iki sinfi ayırmaq qabiliyyətini qiymətləndirir. Log Loss isə proqnozlaşdırılan ehtimalların keyfiyyətini ölçür. ROC-AUC nə qədər yüksək, Log Loss nə qədər aşağı olsa, nəticə bir o qədər yaxşı hesab olunur.

## Feature Importance

CatBoost modelini qurduqdan sonra xüsusiyyətlərin əhəmiyyət göstəricilərinə də baxdım. Məqsədim modelin proqnoz zamanı hansı sütunlardan daha çox istifadə etdiyini görmək idi.

Ən yüksək göstəricilər arasında bunlar oldu:

| Xüsusiyyət | Əhəmiyyət göstəricisi |
|---|---:|
| `channel` | 10.442 |
| `weekday` | 7.084 |
| `publish_dayofyear` | 5.810 |
| `kw_avg_avg` | 5.329 |
| `self_reference_min_shares` | 4.872 |
| `kw_min_avg` | 3.680 |
| `LDA_00` | 3.407 |
| `self_reference_avg_shares` | 3.381 |
| `kw_max_avg` | 3.071 |
| `n_unique_tokens` | 2.524 |

Bu nəticələrdə `channel`, `weekday` və `publish_dayofyear` ilk sıralarda idi. Yəni qurduğum CatBoost modelində bu xüsusiyyətlərin əhəmiyyəti daha yüksək olub.

Buradakı göstəricilər modelin xüsusiyyətlərdən nə dərəcədə istifadə etdiyini göstərir. Bu nəticələri həmin xüsusiyyətlərin hədəf dəyişəninə birbaşa səbəb olduğu kimi şərh etmək olmaz.

## Ensemble Learning

Fərdi modelləri müqayisə etdikdən sonra onların proqnozlarını birləşdirməyi də sınadım. Məqsədim müxtəlif modellərin nəticələrindən birlikdə istifadə etməklə validation göstəricilərini yaxşılaşdırmaq idi.

Əvvəlcə CatBoost və LightGBM modellərini birləşdirdim. CatBoost-a 70%, LightGBM-ə isə 30% çəki verdiyim variantda ROC-AUC 0.75033, Log Loss isə 0.59958 oldu.

Daha sonra XGBoost-u da əlavə etdim və üç modelin müxtəlif çəkilərlə birləşdirilməsini yoxladım.

Sınaqdan keçirdiyim kombinasiyalardan biri belə idi:

- CatBoost - 60%
- LightGBM - 29%
- XGBoost - 11%

Bu variantın validation nəticələri:

| Metrika | Nəticə |
|---|---:|
| ROC-AUC | 0.75038 |
| Log Loss | 0.59966 |

Modellərin proqnozlarını aşağıdakı çəkili orta düsturu ilə birləşdirdim:

```python
P_final = (
    0.60 * P_CatBoost +
    0.29 * P_LightGBM +
    0.11 * P_XGBoost
)
```

Burada hər model `target = 1` üçün ehtimal hesablayır, sonra bu ehtimallar seçilmiş çəkilərlə birləşdirilir.

Müxtəlif çəki kombinasiyalarını da yoxladım. Nəticələr bir-birinə yaxın idi, amma üç modeldən istifadə etdiyim variantda ROC-AUC 0.75038 oldu. Buna görə ensemble yanaşmasını da layihəyə daxil etdim.

## Kaggle submission nəticəsi

CatBoost modeli ilə hazırladığım əvvəlki submission üçün Kaggle-da aşağıdakı balları əldə etmişdim:

- Public Score: 0.75181
- Private Score: 0.73331

Həmin submission-un statusu "Complete (after deadline)" kimi göstərilmişdi. Buna görə bu balları rəsmi yarışma sıralamasında qəbul olunmuş nəticə kimi təqdim etmirəm.

60/29/11 ensemble kombinasiyası üçün qeyd etdiyim göstəricilər isə lokal validation nəticələridir. Bu variantın ayrıca Kaggle balı təsdiqlənmədiyi üçün onu yarışmanın nəticəsi kimi göstərmirəm.

## İstifadə etdiyim texnologiyalar

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost
- Google Colab
- Kaggle

## Layihə zamanı gördüyüm işlər

1. Train və test datasetlərini yoxladım.
2. Çatışmayan dəyərləri və təkrarlanan sətirləri araşdırdım.
3. `target` dəyişəninin paylanmasına baxdım.
4. Tarix sütunlarından yeni xüsusiyyətlər yaratdım.
5. EDA apardım və rəqəmsal xüsusiyyətləri araşdırdım.
6. Əlavə feature engineering variantlarını sınadım.
7. Zaman əsaslı train və validation bölgüsü hazırladım.
8. Logistic Regression, Random Forest, Gradient Boosting, CatBoost, LightGBM və XGBoost modellərini yoxladım.
9. ROC-AUC və Log Loss nəticələrini müqayisə etdim.
10. CatBoost üzrə feature importance təhlili apardım.
11. CatBoost və LightGBM modellərini birləşdirdim.
12. Daha sonra XGBoost-u da əlavə edərək ensemble variantlarını sınaqdan keçirdim.
13. Kaggle üçün submission faylı hazırladım.

## Öyrəndiklərim

Bu layihə mənə Data Science prosesinin müxtəlif mərhələlərində praktiki işləmək imkanı verdi. Datasetin yoxlanılması, EDA, yeni xüsusiyyətlərin yaradılması, model qurulması və nəticələrin müqayisəsi ilə məşğul oldum.

İş zamanı başa düşdüm ki, yalnız daha mürəkkəb model seçmək və ya daha çox xüsusiyyət əlavə etmək nəticəni mütləq yaxşılaşdırmır. Buna görə hər dəyişiklikdən sonra validation göstəricilərini yoxlamağa çalışdım.

Həmçinin bir neçə modeli ayrı-ayrılıqda qiymətləndirdim, sonra onların proqnozlarını birləşdirməyi sınadım. Bu prosesdə fərqli metrikaların nə göstərdiyini və model seçərkən onları birlikdə nəzərə almağın vacibliyini öyrəndim.

Bu repository-də layihə üzərində işləyərkən istifadə etdiyim kodları və əldə etdiyim nəticələri paylaşmışam.
