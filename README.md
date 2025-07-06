

# Product Recommendation System Using LSTM

## Introduction
This project implements a **Product Recommendation System** using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) architecture, designed to provide personalized product suggestions based on user interaction sequences. The system leverages the Retailrocket e-commerce dataset to predict the next item a user is likely to interact with, based on their historical "addtocart" and "transaction" events. The project includes a **production-ready Streamlit web application** for interactive recommendations, making it accessible for users to explore suggestions by selecting visitor IDs or inputting custom item sequences.

### Unique Features
- **Dual-Input LSTM Model**: Combines item IDs and category IDs as inputs to capture both product-specific and categorical patterns, enhancing recommendation relevance.
- **Bidirectional LSTM Variant**: Incorporates a bidirectional LSTM model to capture contextual dependencies in both forward and backward directions, improving sequence understanding.
- **Production-Ready Streamlit App**: A user-friendly web interface allows naive users to explore recommendations via visitor ID selection, search, or manual item sequence input, complete with visualizations of confidence scores.
- **Outlier Handling and Sequence Optimization**: Filters sequences with 3–28 items to ensure meaningful patterns, with outlier detection using IQR for robust preprocessing.
- **Custom Evaluation Metrics**: Uses Precision@5 and Recall@5 to evaluate recommendation quality, tailored for e-commerce scenarios with large item vocabularies.

## Dataset
The system is built using the **Retailrocket e-commerce dataset**, which includes user interactions with products. The dataset is split into multiple files:

- **events.csv**: Contains user interaction data with the following columns:
  - `timestamp`: Time of the event (Unix timestamp).
  - `visitorid`: Unique identifier for the user.
  - `event`: Type of interaction (`view`, `addtocart`, `transaction`).
  - `itemid`: Unique identifier for the product.
  - `transactionid`: Identifier for transactions (populated only for `transaction` events).
- **item_properties_part1.csv** and **item_properties_part2.csv**: Contain item metadata, including:
  - `itemid`: Product identifier.
  - `property`: Property type (e.g., category).
  - `value`: Property value (e.g., category ID).
- **category_tree.csv**: Defines the hierarchical structure of product categories with:
  - `categoryid`: Unique category identifier.
  - `parentid`: Parent category identifier.

The system focuses on `addtocart` and `transaction` events to build meaningful user sequences, filtering out `view` events to prioritize stronger purchase intent.

## Artifacts
The project includes four key files stored in the `artifacts` directory, generated during preprocessing and model training:

1. **final_dataset.pkl**: A pickled Pandas DataFrame containing preprocessed data with user sequences (visitorid, itemid, and category value).
2. **rnn_recommender_optimized.h5**: The trained LSTM model (bidirectional variant) saved in HDF5 format, used for generating recommendations.
3. **item_tokenizer_final.pkl**: A pickled Keras Tokenizer object mapping item IDs to tokenized indices.
4. **category_tokenizer_final.pkl**: A pickled Keras Tokenizer object mapping category IDs to tokenized indices.

These artifacts enable efficient loading of the model and tokenizers for the Streamlit app, ensuring seamless deployment.

## Model Architecture and Results
The project implements four model variants, with results summarized below. The models predict the next item in a sequence based on item and category inputs.

| Model Variant               | Precision@5 | Recall@5 | Why This Result?                                                                 |
|-----------------------------|-------------|----------|----------------------------------------------------------------------------------|
| **Basic LSTM**              | 0.8462      | 0.8462   | Single LSTM captures sequential patterns but struggles with long-term dependencies due to vanishing gradients. |
| **Bidirectional LSTM**      | **0.8535**  | **0.8535** | Captures both forward and backward dependencies, improving context understanding and yielding the best performance. |


### Why Bidirectional LSTM Performed Best
The bidirectional LSTM model outperforms the basic LSTM because it processes sequences in both directions, capturing richer contextual information. This is particularly effective for e-commerce sequences where past and future interactions (within the sequence) provide complementary insights. The model uses:
- **Embedding Layers**: For item IDs (vocab size: 16,435) and category IDs (vocab size: 784), with 50-dimensional embeddings.
- **Concatenation**: Combines item and category embeddings for a unified representation.
- **Bidirectional LSTM**: 32 units to capture temporal dependencies.
- **Dropout (0.5)**: Prevents overfitting.
- **Dense Layer**: Outputs probabilities over the item vocabulary with softmax activation.

The Precision@5 and Recall@5 of 0.8535 indicate that the model correctly predicts the next item in the top-5 recommendations 85.35% of the time, a strong result given the large item vocabulary.

## Streamlit App
The **Streamlit web application** (`main.py`) provides a production-ready interface for users to interact with the recommendation system. Key features include:
- **Input Options**:
  - **Select Visitor ID**: Choose from a dropdown of sampled visitor IDs (3–28 interactions).
  - **Search Visitor ID**: Search for specific visitor IDs with autocomplete suggestions.
  - **Enter Item Sequence**: Input custom item IDs for tailored recommendations.
- **Output**: Displays top-5 recommended products with item IDs, category IDs, and normalized confidence scores.
- **Visualization**: Bar plot of confidence scores for intuitive understanding.
- **Caching**: Uses Streamlit’s caching (`@st.cache_resource`, `@st.cache_data`) for efficient data and model loading.
- **User Guidance**: Includes an expandable section explaining the app’s purpose and usage for naive users.

The app loads the preprocessed dataset, model, and tokenizers from the `artifacts` directory, ensuring fast and reliable recommendations.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vraj-Data-Scientist/product-recommendation-system-using-LSTM
   cd product-recommendation-system-using-LSTM
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `scikit-learn`, `streamlit`.
3. **Download the Dataset**:
   - Obtain the Retailrocket dataset from [Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset).
   - Place `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, and `category_tree.csv` in the project root or update paths in the notebook.
4. **Run the Notebook**:
   - Open `product-recommendation-system-using-RNN (5).ipynb` in Jupyter Notebook.
   - Execute cells to preprocess data, train models, and generate artifacts.
5. **Run the Streamlit App**:
   ```bash
   streamlit run main.py
   ```
   - Access the app at `http://localhost:8501`.
   - Ensure the `artifacts` directory contains the required files.

## Project Structure
```
├── artifacts/
│   ├── final_dataset.pkl
│   ├── rnn_recommender_optimized.h5
│   ├── item_tokenizer_final.pkl
│   ├── category_tokenizer_final.pkl
├── product-recommendation-system-using-RNN (5).ipynb
├── main.py
├── README.md
├── requirements.txt
```

## Future Improvements
- **Attention Mechanism**: Add an attention layer to focus on relevant parts of the sequence.
- **Incorporate Item Properties**: Use additional item metadata (e.g., price, brand) to enhance recommendations.
- **Real-Time Updates**: Integrate real-time user interaction data for dynamic recommendations.
- **Model Optimization**: Experiment with larger LSTM units or deeper architectures for improved accuracy.

## References
- [Retailrocket Dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset)
- [GitHub Repository](https://github.com/Vraj-Data-Scientist/product-recommendation-system-using-LSTM)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

