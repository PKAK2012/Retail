

# 🧠 Deep Learning Based Product Reordering Prediction

This project predicts whether a customer will reorder a product using their historical purchase data. The dataset is based on Instacart's Market Basket Analysis and applies a deep learning model to learn reorder behavior.


## 📊 Dataset Used

Instacart Market Basket Analysis, including:

- `orders.csv`: Order-level info (order number, days since prior order)
- `order_products__train.csv`: Reorder labels for training set
- `order_products__prior.csv`: Historical product purchases
- `products.csv`: Product metadata
- `departments.csv`: Department metadata


## 🔧 Preprocessing Steps

- Merged all datasets to form a comprehensive user-product interaction table.
- Handled missing values (`days_since_prior_order` filled with 0).
- Encoded categorical columns using `LabelEncoder`.
- Scaled numerical features:
  - StandardScaler for continuous features
  - MinMaxScaler for rank-based features


## 📌 Features Used

- `user_id`, `product_id`, `order_number`, `order_dow`, `order_hour_of_day`
- `days_since_prior_order`, `add_to_cart_order`
- `aisle_id`, `department_id`

> ⚠️ Feature engineering was minimal — raw transactional features were used. Can be extended for further improvement.


## 🧠 Model Architecture (DNN)

- Input Layer
- Dense (512) + ReLU + BatchNormalization + Dropout(0.4)
- Dense (256) + ReLU + BatchNormalization + Dropout(0.3)
- Dense (128) + ReLU + BatchNormalization
- Dense (64) + ReLU
- Output Layer: Dense (2) with Softmax

**Loss Function**: Categorical Crossentropy  
**Optimizer**: Adam (LR = 0.001)  
**Callbacks**: EarlyStopping, ReduceLROnPlateau


## 🧪 Evaluation Metrics

Evaluated using:
- Accuracy
- Precision
- Recall
- AUC
- F1-Score
- Confusion Matrix

### ✅ Final Results on Validation Set

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | ~0.67     |
| Precision     | ~0.68     |
| Recall        | ~0.67     |
| AUC           | ~0.73     |
| F1-Score (Weighted) | ~0.65 |
| F1-Score (Reordered class) | **0.76 ✅** |

> ✔️ F1-score for the "Reordered" class exceeds the 0.65 benchmark.


 
## 📦 Model Saving

The trained model is saved using `pickle` for deployment:

```python
import pickle
pickle.dump(model, open("Final.pkl", "wb"))
