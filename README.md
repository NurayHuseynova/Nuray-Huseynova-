# Nuray-Huseynova-
# CTRL IS HERS - Data Science Project

## Layihə haqqında
Bu layihə üzərində Kaggle platformasında keçirilən CTRL IS HERS Final yarışması üçün işləmişəm. Məqsədim datasetlə tanış olmaq, məlumatları təhlil etmək və `target` dəyişənini proqnozlaşdırmaq üçün müxtəlif Machine Learning modellərini sınamaq idi.
İşə datasetin yoxlanılması ilə başladım. Daha sonra məlumatlar üzərində analiz apardım, tarix sütunlarından yeni xüsusiyyətlər yaratdım, bir neçə model qurub nəticələrini müqayisə etdim. Son mərhələdə isə ayrı-ayrı modellərin proqnozlarını birləşdirməyi sınadım.
Layihəni Python və Google Colab istifadə edərək hazırlamışam.
## Dataset
Kaggle yarışmasında təqdim olunan train və test fayllarından istifadə etmişəm.

| Göstərici | Dəyər |
|---|---:|
| Train datası | 31 715 sətir, 49 sütun |
| Test datası | 7 929 sətir, 48 sütun |
| `target = 0` | 14 358 (45,27%) |
| `target = 1` | 17 357 (54,73%) |
| Çatışmayan dəyərlər | Yoxdur |
| Təkrarlanan sətirlər | yoxdur |

`target` dəyişəni 0 və 1 qiymətlərindən ibarətdir. Train datasında 1-lərin sayı bir qədər çox olsa da, siniflər arasında ciddi fərq yoxdur.

Tarix məlumatlarını da yoxlamışam. Train datası əsasən 2013-cü ilin yanvarından 2014-cü ilin avqustuna, test datası isə 2014-cü ilin avqustundan dekabrına qədər olan dövrü əhatə edir.
## Məlumatların hazırlanması və analizi
İlk mərhələdə datasetin ölçüsünü, sütunlarını, məlumat tiplərini, boş dəyərləri və təkrarlanan sətirləri yoxladım. Sonra `target` dəyişəninin paylanmasına və digər xüsusiyyətlərə baxdım.

Tarix sütunundan aşağıdakı xüsusiyyətləri yaratdım:

- `publish_year`
- `publish_month`
- `publish_day`
- `publish_dayofyear`

`weekday` və `channel` kimi kateqorik sütunların modellər üçün uyğun formada işlənməsinə də diqqət yetirdim.

Rəqəmsal sütunları araşdırarkən bəzi xüsusiyyətlərdə qiymətlərin çox qeyri-bərabər paylandığını gördüm. Buna misal olaraq `kw_max_min`, `kw_avg_min` və `self_reference_min_shares` sütunlarını göstərmək olar.

EDA zamanı hədəf dəyişəninin paylanmasını, tarixlə bağlı xüsusiyyətləri, rəqəmsal dəyişənləri və model üçün faydalı ola biləcək göstəriciləri araşdırdım.

## Feature Engineering

Modelin nəticəsini yaxşılaşdırmaq üçün yeni xüsusiyyətlər də sınaqdan keçirdim. Müxtəlif nisbət göstəriciləri, sentiment göstəriciləri arasındakı fərqlər və tarix üçün dövri xüsusiyyətlər əlavə etdim.

Ancaq yeni xüsusiyyətlərin əlavə olunması nəticəni yaxşılaşdırmadı. Həmin sınaqda ROC-AUC 0.74926, Log Loss isə 0.60495 oldu. Əvvəlki modelin nəticələri daha yaxşı olduğuna görə əlavə xüsusiyyətləri yekun variantda istifadə etməməyə qərar verdim.

Bu sınaq göstərdi ki, xüsusiyyətlərin sayını artırmaq həmişə daha yaxşı nəticə vermir. Buna görə dəyişiklikləri validation nəticələrinə əsasən qiymətləndirdim.

## Train və validation

Məlumatları bölərkən zaman ardıcıllığını nəzərə aldım. Təsadüfi bölgü əvəzinə əvvəlki tarixlərdən olan məlumatlarla modeli öyrədib, daha sonrakı tarixlərdə yoxladım.

Bölgüdə təxminən 24 479 sətir train, 7 236 sətir isə validation üçün istifadə olundu.

Bu üsulu seçməyimin səbəbi modelin gələcək tarixlərə aid məlumatlarda necə işlədiyini daha real qiymətləndirmək idi.

## Sınadığım modellər

İlk olaraq Logistic Regression ilə başladım. Daha sonra Random Forest, Gradient Boosting, CatBoost və LightGBM modellərini yoxladım. XGBoost-u da ayrıca model kimi sınaqdan keçirərək sonradan ensemble mərhələsində istifadə etdim.

| Model | ROC-AUC | Log Loss |
|---|---:|---:|
| Logistic Regression | 0.7291 | 0.6178 |
| Random Forest | 0.7360 | 0.6094 |
| Gradient Boosting | 0.7432 | 0.6051 |
| CatBoost | 0.7497 | 0.6005 |
| LightGBM | 0.7477 | 0.6001 |

Logistic Regression üçün digər göstəricilər də belə oldu:

| Metrika | Nəticə |
|---|---:|
| Accuracy | 0.6797 |
| Precision | 0.7197 |
| Recall | 0.6767 |
| F1-score | 0.6975 |
| ROC-AUC | 0.7291 |
| Log Loss | 0.6178 |

Bu sınaqlarda CatBoost ən yüksək fərdi ROC-AUC nəticəsini göstərdi. LightGBM isə bir qədər aşağı Log Loss verdi.

Modelləri müqayisə edərkən yalnız bir göstəriciyə əsaslanmadım. ROC-AUC və Log Loss nəticələrini birlikdə nəzərə aldım. ROC-AUC-nin yüksək, Log Loss-un isə aşağı olması üstün hesab olunur.

## Feature Importance

CatBoost modelinin feature importance nəticələrinə baxaraq hansı xüsusiyyətlərin model üçün daha əhəmiyyətli olduğunu araşdırdım.

Ən yüksək göstəricilər arasında bunlar var idi:

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

Nəticələrdə `channel`, `weekday` və `publish_dayofyear` xüsusiyyətləri ön sıralarda idi. Bu, onların qurduğum CatBoost modelinin proqnozlarında əhəmiyyətli rol oynadığını göstərir, lakin birbaşa səbəb-nəticə əlaqəsi demək deyil.

## Ensemble Learning

Fərdi modellərin nəticələrini əldə etdikdən sonra onların proqnozlarını birləşdirməyi sınadım.

Əvvəlcə CatBoost və LightGBM ilə başladım. CatBoost-a 70%, LightGBM-ə isə 30% çəki verdikdə ROC-AUC 0.75033, Log Loss isə 0.59958 oldu.

Daha sonra XGBoost-u da kombinasiyaya əlavə etdim. Sınaqdan keçirdiyim variantlardan biri aşağıdakı kimi idi:

- CatBoost - 60%
- LightGBM - 29%
- XGBoost - 11%

Bu kombinasiyanın validation nəticələri:

| Metrika | Nəticə |
|---|---:|
| ROC-AUC | 0.75038 |
| Log Loss | 0.59966 |

Proqnozların birləşdirilməsi üçün hər modelin `target = 1` ehtimalını götürüb çəkili orta hesabladım:

```python
P_final = (0.60 * P_CatBoost +0.29 * P_LightGBM +0.11 * P_XGBoost)
```

Fərqli çəki kombinasiyalarını da yoxladım. Nəticələr bir-birinə yaxın olduğu üçün fərqi şişirtmək düzgün olmaz. Üç modelin birləşdirilməsi validation nəticəsini bir qədər yaxşılaşdırdı.

## Kaggle submission

CatBoost modeli ilə hazırladığım əvvəlki submission üçün Kaggle-da aşağıdakı ballar göstərildi:

- Public Score: 0.75181
- Private Score: 0.73331

Submission statusu "Complete (after deadline)" olaraq göstərilmişdi. Buna görə bu nəticəni rəsmi yarışma sıralamasında qəbul edilmiş nəticə kimi təqdim etmirəm.

60/29/11 ensemble kombinasiyası üçün göstərilən göstəricilər lokal validation nəticələridir. Onun ayrıca Kaggle nəticəsini əldə etmədən həmin balı yarışma nəticəsi kimi qeyd etmək olmaz.

## İstifadə olunan texnologiyalar

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

## Layihənin mərhələləri

1. Train və test datasının yoxlanılması.
2. Məlumatların ilkin təhlili.
3. Tarix xüsusiyyətlərinin yaradılması.
4. EDA və xüsusiyyətlərin araşdırılması.
5. Müxtəlif Machine Learning modellərinin qurulması.
6. Zaman əsaslı validation və nəticələrin müqayisəsi.
7. Əlavə xüsusiyyətlərin yaradılmasının sınaqdan keçirilməsi.
8. Feature importance təhlili.
9. Modellərin proqnozlarının birləşdirilməsi.
10. Kaggle submission faylının hazırlanması.

## Layihədən qazandığım təcrübə

Bu layihə zamanı datasetin ilkin təhlilindən başlayaraq model qurulmasına və nəticələrin müqayisəsinə qədər müxtəlif mərhələlər üzərində işləmişəm. Logistic Regression, Random Forest, Gradient Boosting, CatBoost, LightGBM və XGBoost modellərini sınamışam.

Ən çox diqqət yetirdiyim məqamlardan biri modelin nəticəsini yalnız train datasında deyil, ayrıca validation datasında qiymətləndirmək oldu. Həmçinin yeni xüsusiyyətləri və ensemble kombinasiyalarını sınaqdan keçirərək dəyişikliklərin həqiqətən faydalı olub-olmadığını yoxladım.

Bu repository-də layihə üzrə istifadə etdiyim kodu və əldə etdiyim nəticələri paylaşıram.
