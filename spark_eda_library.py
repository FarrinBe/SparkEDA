from pyspark.sql import DataFrame, functions as F
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
import matplotlib.pyplot as plt

class SparkEDA:
    def __init__(self, spark_df: DataFrame):
        self.spark_df = spark_df

    def data_overview(self):
        """
        Prints the number of rows, number of columns, and schema of a Spark DataFrame.
        """
        print(f"Number of rows: {self.spark_df.count()}")
        print(f"Number of columns: {len(self.spark_df.columns)}")
        print("Schema:")
        self.spark_df.printSchema()

    def describe_numerical(self):
        """
        Returns a summary statistics of the numerical columns in a Spark DataFrame.
        """
        return self.spark_df.describe().show()

    def describe_categorical(self, categorical_columns):
        """
        Displays unique values and their counts for each categorical column in a Spark DataFrame.
        """
        for column in categorical_columns:
            self.spark_df.select(column).distinct().show()
            self.spark_df.groupBy(column).count().orderBy('count', ascending=False).show()

    def plot_histogram(self, column, bins=20, fraction=0.1):
        """
        Uses Spark to collect data and Matplotlib to plot a histogram of a specified column.
        """
        data = self.spark_df.select(column).rdd.flatMap(lambda x: x).collect()
        #sampled_data = self.spark_df.select(column).sample(fraction)
        
        plt.hist(data, bins)
        plt.title(f'Histogram of {column}')
        plt.show()

    def correlation_matrix(self, input_cols):
        """
        Calculates the Pearson correlation matrix for the specified input columns in a Spark DataFrame.
        """
        vector_col = "features"
        assembler = VectorAssembler(inputCols=input_cols, outputCol=vector_col)
        df_vector = assembler.transform(self.spark_df).select(vector_col)
        
        matrix = Correlation.corr(df_vector, vector_col)
        return matrix.collect()[0]["pearson({})".format(vector_col)].values

    def feature_summary(self):
        """
        Provides a summary of statistical measures and missing value counts for a Spark DataFrame.
        """
        summary = self.spark_df.summary("count", "min", "max", "mean")
        summary.show()
        
        missing_val_counts = self.spark_df.select([F.count(F.col(c).isNull().cast("int")).alias(c) for c in self.spark_df.columns])
        print("Missing Value Counts:")
        missing_val_counts.show()

    def value_counts(self, column):
        """
        Calculates the frequency of each unique value in a specified column of a Spark DataFrame and displays the results in descending order of frequency.
        """
        return self.spark_df.groupBy(column).count().orderBy('count', ascending=False).show()

    def sample_data(self, fraction=0.1):
        """
        Takes a fraction parameter, samples a fraction of the Spark DataFrame, and returns the sampled data as a Pandas DataFrame.
        """
        return self.spark_df.sample(fraction=fraction).toPandas()

    def time_series_plot(self, date_col, value_col):
        """
        Plots a time series graph of the specified value column over the specified date column.
        """
        pd_df = self.spark_df.select(date_col, value_col).toPandas()
        pd_df[date_col] = pd.to_datetime(pd_df[date_col])
        pd_df.set_index(date_col, inplace=True)
        pd_df[value_col].plot(figsize=(10, 6))
        
        # Add title and axis labels
        plt.title(f'Time Series Plot of {value_col} over {date_col}')
        plt.xlabel(date_col)
        plt.ylabel(value_col)
        plt.show()
        
    def pairwise_scatter_plot(self, cols):
        """
        Generates pairwise scatter plots for the specified list of columns.
        Note: Due to Spark's nature, data collection to the driver node is required.
        This should be used with caution on large datasets.
        """
        pd_df = self.spark_df.select(cols).toPandas()
        pd.plotting.scatter_matrix(pd_df, figsize=(12, 12))
        
    def feature_importance(self, feature_cols, label_col, importance_type='gain'):
        """
        Estimates feature importance using a simple model (e.g., Decision Tree).
        This method can give insights into which features might be most predictive of the outcome.
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        assembled_df = assembler.transform(self.spark_df)

        dt = DecisionTreeClassifier(labelCol=label_col, featuresCol="features")
        model = dt.fit(assembled_df)

        importances = model.featureImportances
        for i, col in enumerate(feature_cols):
            print(f"{col}: {importances[i]}")
            
    def fill_missing(self, strategy='mean', columns=None):
        """
        Fills missing values in specified columns according to the given strategy ('mean', 'median', 'mode', or a constant value).
        If columns are not specified, applies to all applicable columns.
        """
        if strategy not in ['mean', 'median', 'mode', 'constant']:
            raise ValueError("Strategy not supported. Choose from 'mean', 'median', 'mode', or 'constant'.")

        if not columns:
            columns = self.spark_df.columns

        for col_name in columns:
            if strategy == 'mean':
                mean_value = self.spark_df.select(F.mean(F.col(col_name)).alias('mean')).collect()[0]['mean']
                self.spark_df = self.spark_df.na.fill({col_name: mean_value})
            elif strategy == 'median':
                # Median can be more complex in Spark, involving sorting and taking the middle value. Resource intensive.
                pass
            elif strategy == 'mode':
                mode_value = self.spark_df.groupBy(col_name).count().orderBy('count', ascending=False).first()[0]
                self.spark_df = self.spark_df.na.fill({col_name: mode_value})
            elif strategy == 'constant':
                self.spark_df = self.spark_df.na.fill({col_name: 0})  # Example: filling with 0

        return self.spark_df





# Example usage:
# spark_df = <your_spark_dataframe_here>
# eda = SparkEDA(spark_df)
# eda.data_overview()
# eda.describe_numerical()
# ... and so on for other methods
