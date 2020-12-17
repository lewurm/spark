/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example for Multilayer Perceptron Classification.
 */
object MultilayerLewurmExample {
  def time[R](block: => R, desc: String): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    val elapsed = t1 - t0
    println(s"Elapsed time for $desc: ${math.round(elapsed/1000000000.0)}s (${elapsed}ns)")
    result
  }

  def main(args: Array[String]): Unit = {
    val num_of_lcores = Runtime.getRuntime().availableProcessors()
    // val sc = new SparkContext(s"local[$num_of_lcores]", "MultilayerLewurmBench",
    //     sys.env("SPARK_HOME"),
    //     List("target/scala-2.12/classification-benchmark_2.12-1.0.jar"));
    val spark = SparkSession.builder
      .master(s"local[$num_of_lcores]")
      .appName("MultilayerLewurmExample")
      .getOrCreate()

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.8, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    train.printSchema()

    val head = train.head()
    println(s"train head: $head")

    // specify layers for the neural network. This will increase computation time.
    val layers = Array[Int](4, 1000, 500, 250, 125, 62, 31, 16, 8, 3)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setTol(Double.MinPositiveValue) // minimal tolerance, increases computation time
      .setMaxIter(100*4000)

    // train the model
    val model = time({ trainer.fit(train) }, "Model Training")

    // compute accuracy on the test set
    val result = time({ model.transform(test) }, "Model Test")
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
