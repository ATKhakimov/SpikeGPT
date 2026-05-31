**УДК 004.8:537.312 DOI:**

**Прогнозирование деградации мемристоров в условиях сдвига домена с использованием transfer learning и моделей временных рядов**

**Алямовская А. А.<sup>1,2</sup>, Мещанинов Ф. П.<sup>2</sup>, Жевненко Д. А.<sup>2</sup>, Горнев Е. С. <sup>1,2</sup>**

_<sup>1</sup> Национальный исследовательский университет «Московский физико-технический институт»,_

_141701, Московская область, г. Долгопрудный, Институтский переулок, д. 9._

_<sup>2</sup> АО «НИИМЭ»_

_124460, Россия, Москва, Зеленоград, улица Академика Валиева, 6/1_

[_aalyamovskaya@niime.ru_](mailto:tpavlova@niime.ru)

**Аннотация**

В работе исследуется задача прогнозирования поведения мемристоров при изменении частоты и режима работы (сдвиг домена). Сравниваются современные модели временных рядов и подходы transfer learning, включая полный и частичный fine-tuning. Эксперименты на открытых бенчмарках и реальном датасете ННГУ показывают, что модели, в архитектуре которых явно используется декомпозиция на тренд и сезонность показывают наилучшее качество, превосходя более сложные методы более чем на 20%.

**Ключевые слова:** мемристоры; временной ряд; машинное обучение; трансфер лернинг.

Мемристоры являются ключевыми элементами нейроморфных вычислительных систем благодаря их способности хранить состояние и моделировать синаптическую пластичность \[1\]. Однако из-за сложной динамики деградации и чувствительности к условиям эксплуатации применение мемристоров на данный момент ограничено. Характеристики мемристоров зависят как от частоты переключения и режима работы, так и от внешних факторов. В результате возникает необходимость в решении задачи сдвига домена, когда меняется чистота сигнала или режим работы устройства. Классические методы анализа временных рядов, такие как ARIMA \[2\] и SARIMA \[3\], а также такие нейросетевые подходы, как CNN и RNN, демонстрируют слабую обобщающую способность при смене домена, а также низкую эффективность при долгосрочном прогнозировании временных рядов.

Целью данной работы является разработка и оценка моделей временных рядов для устойчивого прогнозирования поведения мемристоров при изменении частоты и режима работы. В рамках данного исследования рассматриваются такие модели, как TimeMixer \[4\], DLinear \[5\], и PatchTST \[6\], охватывающие основные классы архитектур долгосрочного предсказания временных рядов. Первичная валидация моделей производится на открытых бенчмарках для прогнозирования временных рядов: ETT \[7\] (температурные ряды трансформаторов с различной частотой дискретизации), Electricity \[8\] (временные ряды электропотребления) и C-MAPSS \[9\] (синтетические данные деградации авиационных двигателей), что позволяет оценить обобщающую способность моделей при работе с различными типами данных.

Данное исследование включает такие этапы, как анализ робастности моделей при обучении и тестировании на зашумленных версиях датасетов ETT и Electricity, исследование междоменного zero-shot переноса внутри датасетов C-MAPSS и ETT, а также рассмотрение различных подходов transfer learning. В работе рассматриваются следующие методики проведения трансфера: протокол TL-1 (warm-start с последующим дообучением всей модели) и протокол TL-2 (head-only finetune), при котором основная часть модели замораживается, а обучение проводится только для выходной головы. Последний подход направлен на снижение вероятности переобучения при ограниченном объеме данных.

Основная часть эксперимента проведена на данных реального исследования от ННГУ для ячейки памяти WU6, расположенной на тестовом кристалле 7NIIIS1 со структурой Au(40nm)/Ta(40nm)/Al<sub>2</sub>O<sub>3</sub>(6 nm)/ZrO<sub>2</sub>(12%-Y)(20nm)/Pt(40nm)/Ti(10nm). Эксперимент включал запись вольтамперных характеристик с приложением последовательности импульсов треугольной формы с частотами 25, 50 и 100 кГц.

Полученные результаты показывают, что модели с явной декомпозицией временного ряда обладают наилучшей устойчивостью к сдвигу домена и обеспечивают высокое качество прогнозирования временных рядов мемристоров. Для модели DLinear на проприетарном датасете ННГУ достигнуты значения MSE = (0.0036 ± 0.0002) и MAE = (0.03452 ± 0.0015), что более чем на 20% лучше по метрике MAE по сравнению как с базовыми, так и SOTA архитектурами, рассмотренными в данной работе. При этом модель демонстрирует устойчивость к увеличению горизонта прогноза (деградация менее 4% при добавлении одного цикла переключения) и сохраняет работоспособность в условиях межчастотного zero-shot переноса (25 -> 100 кГц) с умеренной деградацией качества (менее 20 процентов по метрике MSE), что подтверждает эффективность простых интерпретируемых подходов для задач прогнозирования деградации мемристоров.

**_Литература_**

- Гриценко В. А., Гисматулин А. А., Орлов О. М. Запоминающие свойства мемристоров на основе оксида и нитрида кремния //Российские нанотехнологии. 2021. Т. 16. №. 6. С. 751-760.
- Ho S. L., Xie M. The use of ARIMA models for reliability forecasting and analysis //Computers & industrial engineering. 1998. V. 35. №. 1-2. С. 213-216.
- Suhartono S. Time series forecasting by using seasonal autoregressive integrated moving average: Subset, multiplicative or additive model //Journal of Mathematics and Statistics. 2011. V. 7. №. 1. С. 20-27.
- Wang S. \[et al.\] Timemixer: Decomposable multiscale mixing for time series forecasting //arXiv preprint arXiv:2405.14616. 2024.
- Zeng A. \[et al.\] Are transformers effective for time series forecasting? //Proceedings of the AAAI conference on artificial intelligence. 2023. V. 37. №. 9. С. 11121-11128.
- Nie Y. \[et al.\] A time series is worth 64 words: Long-term forecasting with transformers //arXiv preprint arXiv:2211.14730. 2022.
- Zhou H. \[et al.\] Informer: Beyond efficient transformer for long sequence time-series forecasting //Proceedings of the AAAI conference on artificial intelligence. 2021. V. 35. №. 12. С. 11106-11115.
- URL: <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014> (дата обращения: 19.02.2026).
- Ramasso E., Saxena A. Review and analysis of algorithmic approaches developed for prognostics on CMAPSS dataset //Annual Conference of the Prognostics and Health Management Society 2014. 2014.

**Predicting memristor degradation under domain-shifting conditions using transfer learning and time-series models**

**Alyamovskaya A. A.<sup>1,2</sup>, Meshchaninov F. P. <sup>2</sup>, Zhevnenko E. A.<sup>2</sup>, Gornev E. S.<sup>1,2</sup>**

_<sup>1</sup> Moscow Institute of Physics and Technology_

_141701, Moscow region, Dolgoprudny, Institutskiy pereulok, 9._

_<sup>2</sup> JSC «MERI»_

_124460, Russia, Moscow, Zelenograd, Akademik Valiev Street, 6/1_

[_aalyamovskaya@niime.ru_](mailto:aalyamovskaya@niime.ru)

**Abstract**

This paper investigates the problem of predicting the behaviour of memristors when the frequency and operating mode change (domain shift). Modern time series models and transfer learning approaches, including full and partial fine-tuning, are compared. Experiments on open benchmarks and the real-world NNU dataset show that models whose architecture explicitly utilises decomposition into trend and seasonality demonstrate the best performance, outperforming more complex methods by over 20%.

**Keywords:** memristors; time series; machine learning; transfer learning.

Memristors are key elements of neuromorphic computing systems due to their ability to store state and model synaptic plasticity \[1\]. However, due to complex degradation dynamics and sensitivity to operating conditions, the application of memristors is currently limited. The characteristics of memristors depend on both switching frequency and operating mode, as well as external factors. Consequently, there is a need to address the problem of domain shift when signal purity or the device's operating mode changes. Classical time series analysis methods, such as ARIMA \[2\] and SARIMA \[3\], as well as neural network approaches such as CNNs and RNNs, demonstrate poor generalisation ability when the domain changes, as well as low efficiency in long-term time series forecasting.

The aim of this work is to develop and evaluate time series models for robust forecasting of memristor behaviour under changes in frequency and operating mode. This study examines models such as TimeMixer \[4\], DLinear \[5\], and PatchTST \[6\], covering the main classes of architectures for long-term time series prediction. Initial validation of the models is performed on open time series forecasting benchmarks: ETT \[7\] (temperature time series of transformers with varying sampling rates), Electricity \[8\] (electricity consumption time series) and C-MAPSS \[9\] (synthetic data on aircraft engine degradation), which allows the generalisation ability of the models to be assessed when working with different types of data.

This study includes stages such as analysing the robustness of models during training and testing on noisy versions of the ETT and Electricity datasets, investigating cross-domain zero-shot transfer within the C-MAPSS and ETT datasets, and examining various transfer learning approaches. The paper examines the following transfer methods: the TL-1 protocol (warm-start followed by fine-tuning of the entire model) and the TL-2 protocol (head-only fine-tuning), in which the main part of the model is frozen and training is performed only on the output head. The latter approach aims to reduce the likelihood of overfitting when data is limited.

The main part of the experiment was conducted using real-world research data from NNU for the WU6 memory cell. The experiment involved recording current-voltage characteristics whilst applying a sequence of triangular pulses at frequencies of 25, 50 and 100 kHz.

The results show that models with explicit time series decomposition exhibit the best resistance to domain shift and provide high-quality forecasting of memristor time series. For the DLinear model on the proprietary NNU dataset, values of MSE = (0.0036 ± 0.0002) and MAE = (0.03452 ± 0.0015) were achieved on the proprietary NNU dataset, which is more than 20% better on the MAE metric compared to both the baseline and SOTA architectures considered in this work. Furthermore, the model demonstrates robustness to an increase in the forecast horizon (less than 4% degradation when adding one switching cycle) and remains operational under zero-shot inter-frequency transfer (25 -> 100 kHz) with moderate quality degradation (less than 20% on the MSE metric), confirming the effectiveness of simple, interpretable approaches for memristor degradation prediction tasks.

**_References_**

- Gritsenko V. A., Gismatulin A. A., Orlov O. M. Memory properties of silicon oxide and nitride-based memristors //Russian Nanotechnologies. 2021. Vol. 16. No. 6. pp. 751-760.
- Ho S. L., Xie M. The use of ARIMA models for reliability forecasting and analysis //Computers & industrial engineering. 1998. V. 35. №. 1-2. С. 213-216.
- Suhartono S. Time series forecasting by using seasonal autoregressive integrated moving average: Subset, multiplicative or additive model //Journal of Mathematics and Statistics. 2011. V. 7. №. 1. С. 20-27.
- Wang S. \[et al.\] Timemixer: Decomposable multiscale mixing for time series forecasting //arXiv preprint arXiv:2405.14616. 2024.
- Zeng A. \[et al.\] Are transformers effective for time series forecasting? //Proceedings of the AAAI conference on artificial intelligence. 2023. V. 37. №. 9. С. 11121-11128.
- Nie Y. \[et al.\] A time series is worth 64 words: Long-term forecasting with transformers //arXiv preprint arXiv:2211.14730. 2022.
- Zhou H. \[et al.\] Informer: Beyond efficient transformer for long sequence time-series forecasting //Proceedings of the AAAI conference on artificial intelligence. 2021. V. 35. №. 12. С. 11106-11115.
- URL: <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014> (дата обращения: 19.02.2026).
- Ramasso E., Saxena A. Review and analysis of algorithmic approaches developed for prognostics on CMAPSS dataset //Annual Conference of the Prognostics and Health Management Society 2014. 2014.