package com.asynctaskcoffee.cnngenderestimate.CNNDetectors;


import android.os.Environment;
import android.util.Log;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.io.File;

import static org.bytedeco.opencv.global.opencv_core.NORM_MINMAX;
import static org.bytedeco.opencv.global.opencv_core.normalize;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;


public class GenderEstimator {

    private Net genderNet;

    public GenderEstimator() {
        try {
            genderNet = new Net();

            File dir = Environment.getExternalStorageDirectory();
            File protobuf = new File(dir, "/CNNFiles/deploy_gendernet.prototxt");

            File dirTwo = Environment.getExternalStorageDirectory();
            File caffeModel = new File(dirTwo, "/CNNFiles/gender_net.caffemodel");


            if (caffeModel.exists()) {
                Log.e("coffe model", "exists");
            }

            if (protobuf.exists()) {
                Log.e("protobuf", "exists");
            }

            genderNet = readNetFromCaffe(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public String predictGender(Mat face, Frame frame) {
        try {
            Mat croppedMat = new Mat();
            resize(face, croppedMat, new Size(256, 256));
            normalize(croppedMat, croppedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);

            Mat inputBlob = blobFromImage(croppedMat);
            genderNet.setInput(inputBlob, "data", 1.0, null);

            Mat prob = genderNet.forward("prob");

            Indexer indexer = prob.createIndexer();

            if (indexer.getDouble(0, 0) > indexer.getDouble(0, 1)) {
                return "ERKEK:  " + indexer.toString();
            } else {
                return "KADIN:  " + indexer.toString();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "Sonnuçlandırılamadı";
    }

}