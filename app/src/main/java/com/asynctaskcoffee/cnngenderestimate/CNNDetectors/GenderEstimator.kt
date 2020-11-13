package com.asynctaskcoffee.cnngenderestimate.CNNDetectors

import android.content.Context
import org.bytedeco.javacpp.indexer.Indexer
import org.bytedeco.javacv.Frame
import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.global.opencv_dnn
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Size
import org.bytedeco.opencv.opencv_dnn.Net
import java.io.File
import java.io.IOException
import kotlin.math.pow

class GenderEstimator(context: Context) {
    private var genderNet: Net? = null
    fun predictGender(face: Mat?, frame: Frame): String {
        try {
            val croppedMat = Mat()
            opencv_imgproc.resize(face, croppedMat, Size(256, 256))
            opencv_core.normalize(croppedMat, croppedMat, 0.0, 2.0.pow(frame.imageDepth.toDouble()), opencv_core.NORM_MINMAX, -1, null)
            val inputBlob = opencv_dnn.blobFromImage(croppedMat)
            genderNet!!.setInput(inputBlob, "data", 1.0, null)
            val prob = genderNet!!.forward("prob")
            val indexer = prob.createIndexer<Indexer>()
            return if (indexer.getDouble(0, 0) > indexer.getDouble(0, 1)) {
                "MAN:  $indexer"
            } else {
                "WOMAN:  $indexer"
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return "Could not be determined"
    }

    init {
        try {
            genderNet = Net()
            genderNet = opencv_dnn.readNetFromCaffe(getFileFromAssets(context, "deploy_gendernet.prototxt").absolutePath, getFileFromAssets(context, "gender_net.caffemodel").absolutePath)
        } catch (e: Exception) {
            e.printStackTrace()
        }
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