package com.asynctaskcoffee.cnngenderestimate;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.asynctaskcoffee.cnngenderestimate.CNNDetectors.AgeEstimator;
import com.asynctaskcoffee.cnngenderestimate.CNNDetectors.GenderEstimator;
import com.esafirm.imagepicker.features.ImagePicker;
import com.esafirm.imagepicker.model.Image;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.opencv.imgproc.Imgproc;

import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class MainActivity extends AppCompatActivity {

    private int REQUEST_CAMERA_PERMISSION = 200;
    private int REQUEST_READ_PERMISSION = 222;
    private GenderEstimator cnnGender;
    private AgeEstimator cnnAge;
    private ImageView imgPerson;
    private TextView sonuc_tv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setViews();
        initPermissionForCNNImplementation();
    }

    private void initPermissionForCNNImplementation() {
        if (checkSelfPermission(READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{READ_EXTERNAL_STORAGE}, REQUEST_READ_PERMISSION);
        } else {
            cnnGender = new GenderEstimator(this);
            cnnAge = new AgeEstimator(this);
        }
    }

    private void setViews() {
        imgPerson = findViewById(R.id.img_person);
        sonuc_tv = findViewById(R.id.sonuc_tv);
    }

    public void fotoCek(View view) {
        if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            startCameraForResult();
        }
    }


    private void startCameraForResult() {
        ImagePicker.cameraOnly().start(this);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCameraForResult();
        }
        if (requestCode == REQUEST_READ_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            cnnGender = new GenderEstimator(this);
            cnnAge = new AgeEstimator(this);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (ImagePicker.shouldHandle(requestCode, resultCode, data)) {
            Image image = ImagePicker.getFirstImageOrNull(data);
            if (image != null) {
                imgPerson.setImageURI(Uri.parse(image.getPath()));
                startFaceDetection(BitmapFactory.decodeFile(image.getPath()));
            }
        }
    }


    @SuppressLint("SetTextI18n")
    private void startFaceDetection(Bitmap faceBitmap) {
        Frame frame = new AndroidFrameConverter().convert(faceBitmap);
        Mat mat = new OpenCVFrameConverter.ToMat().convert(frame);
        resize(mat, mat, new Size(227, 227));
        cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR);
        sonuc_tv.setText("SEX : " + cnnGender.predictGender(mat, frame) +
                "\n" +
                "\n" +
                "AGE : " + cnnAge.predictAge(mat, frame));
    }

}
