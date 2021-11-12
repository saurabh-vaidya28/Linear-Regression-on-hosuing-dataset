package linear_regression
import org.apache.spark.ml.feature.{Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object linear_regression {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("Missing Demo")
    val sc = new SparkContext(sparkConf)

    val spark = SparkSession.builder().getOrCreate()

    import spark.implicits._
    // Load training data
    var houses = spark.read.option("header", true)
      .csv("C:\\Users\\100_rabh\\IdeaProjects\\Mini_Project(ML)\\" +
        "src\\main\\scala\\linear_regression\\Housing.csv")

    val colNames = Array("price", "area", "bedrooms", "bathrooms", "stories", "parking")
    for (colName <- colNames){
      houses = houses.withColumn(colName, col(colName).cast("Double"))
    }

    println("Housing Dataframe:")
    houses.show(5)

    println("Print Schema:")
    houses.printSchema()

    println("Describing statistical information:")
    houses.select("price", "area", "bedrooms", "bathrooms", "stories", "parking").describe().show()

    val colIndexes = Array("mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea")
    for (colIndex <- colIndexes){
      val indexer = new StringIndexer()
        .setInputCol(colIndex)
        .setOutputCol(colIndex + "Idx")
     houses = indexer.fit(houses).transform(houses)
    }

    // dropping columns with yes/no as boolean value
    val cols = Seq("mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus")
    val houses7 = houses.drop(cols: _*)

    println("Converted Dataframe to 0/1: ")
    houses7.show(5)

    // selecting particular columns which we want to scale for fitting in linear regression
    val columns_to_scale = houses7.select("price", "area", "bedrooms", "bathrooms", "stories", "parking").toDF("price", "area", "bedrooms", "bathrooms", "stories", "parking")
    println("Columns to scale: ")
    columns_to_scale.show(5)

    // making features columns which will contains all the selected columns
    val assembler = new VectorAssembler()
      .setInputCols(Array("area", "bedrooms", "bathrooms", "stories", "parking"))
      .setOutputCol("features")

    val df = assembler.transform(columns_to_scale)
    println("Separate features column which contains the remaining columns: ")
    df.show(5)

    // min max Scaler
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2.0)
      .transform(df)

    println("Normalized features column into normFeatures: ")
    normalizer.show(5)

    // Split the data into training and test sets (30% held out for testing).
    val Array(training, test) = normalizer.randomSplit(Array(0.7, 0.3))

    // Applying linear regression model
    val lr = new LinearRegression()
      .setLabelCol("price")
      .setFeaturesCol("normFeatures")
      .setMaxIter(10)
      .setRegParam(1.0)
      .setElasticNetParam(1.0)

    // Fit the model
    val lrModel = lr.fit(training)
    // print the output column and the input column
    println("Dataframe with input column and output column:")
    lrModel.transform(test)
      .select("features", "normFeatures", "price", "prediction")
      .show()

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} \nIntercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"Total Iterations: ${trainingSummary.totalIterations}")
    println(s"Objective History: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show(5)
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"R2: ${trainingSummary.r2}")
  }
}