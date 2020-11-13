package com.asynctaskcoffee.cnngenderestimate.CNNDetectors;

import android.content.Context
import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacv.Frame
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_dnn.blobFromImage
import org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Point
import org.bytedeco.opencv.opencv_core.Size
import org.bytedeco.opencv.opencv_dnn.Net
import java.io.File
import java.io.IOException
import java.nio.DoubleBuffer
import kotlin.math.pow


class AgeEstimator(context: Context) {

    var AGES = arrayOf("0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-")

    private var ageNet: Net? = null

    init {
        try {
            try {
                ageNet = Net()
                ageNet = readNetFromCaffe(getFileFromAssets(context, "deploy_agenet.prototxt").absolutePath, getFileFromAssets(context, "age_net.caffemodel").absolutePath)
            } catch (e: Exception) {
                e.printStackTrace()
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun predictAge(face: Mat, frame: Frame): String {
        try {
            val resizedMat = Mat()
            resize(face, resizedMat, Size(256, 256))
            normalize(resizedMat, resizedMat, 0.0, 2.0.pow(frame.imageDepth.toDouble()), NORM_MINMAX, -1, null)
            val inputBlob = blobFromImage(resizedMat)
            ageNet?.setInput(inputBlob, "data", 1.0, null)
            val prob = ageNet?.forward("prob")
            val pointer = DoublePointer(DoubleBuffer.allocate(1))
            val max = Point()
            minMaxLoc(prob, null, pointer, null, max, null)
            return AGES[max.x()]
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return "Could not be determined"
    }

    @Throws(IOException::class)
    fun getFileFromAssets(context: Context, fileName: String): File = File(context.cacheDir, fileName)
            .also {
                if (!it.exists()) {
                    it.outputStream().use { cache ->
                        context.assets.open(fileName).use { inputStream ->
                            inputStream.copyTo(cache)
                        }
                    }
                }
            }
}
