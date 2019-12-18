package com.asynctaskcoffee.cnngenderestimate.CNNDetectors;

import android.os.Environment;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.io.File;

import static org.bytedeco.opencv.global.opencv_core.NORM_MINMAX;
import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_core.normalize;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;


public class AgeEstimator {

    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-"};

    private Net ageNet;

    public AgeEstimator() {
        try {
            ageNet = new Net();

            File dir = Environment.getExternalStorageDirectory();
            File protobuf = new File(dir, "/CNNFiles/deploy_agenet.prototxt");

            File dirTwo = Environment.getExternalStorageDirectory();
            File caffeModel = new File(dirTwo, "/CNNFiles/age_net.caffemodel");

            ageNet = readNetFromCaffe(protobuf.getAbsolutePath(), caffeModel.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String predictAge(Mat face, Frame frame) {
        try {
            Mat resizedMat = new Mat();
            resize(face, resizedMat, new Size(256, 256));
            normalize(resizedMat, resizedMat, 0, Math.pow(2, frame.imageDepth), NORM_MINMAX, -1, null);

            Mat inputBlob = blobFromImage(resizedMat);

            ageNet.setInput(inputBlob, "data", 1.0, null);
            Mat prob = ageNet.forward("prob");

            DoublePointer pointer = new DoublePointer(new double[1]);
            Point max = new Point();
            minMaxLoc(prob, null, pointer, null, max, null);

            return AGES[max.x()];

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

}
