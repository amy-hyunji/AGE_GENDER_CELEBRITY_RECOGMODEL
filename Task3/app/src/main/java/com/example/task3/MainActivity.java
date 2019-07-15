package com.example.task3;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.gson.Gson;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import com.github.chrisbanes.photoview.PhotoView;

public class MainActivity extends AppCompatActivity implements OnClickListener {


    public static final String URL = "http://143.248.36.213:3355/";
    
    private static final int INTENT_REQUEST_ALBUM = 100;
    private static final int INTENT_REQUEST_CAMERA = 50;
    
    private PhotoView photoView;
    private TextView textView;
    private String imageString, celebrityString;
    private Animation fab_open, fab_close;
    private Boolean isFabOpen = false;
    private FloatingActionButton fab, fab1, fab2, fab3, fab4;
    String currentPhotoPath;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        fab_open = AnimationUtils.loadAnimation(getApplicationContext(), R.anim.fab_open);
        fab_close = AnimationUtils.loadAnimation(getApplicationContext(), R.anim.fab_close);
        
        fab = (FloatingActionButton) findViewById(R.id.fab);
        fab1 = (FloatingActionButton) findViewById(R.id.fab1);
        fab2 = (FloatingActionButton) findViewById(R.id.fab2);
        fab3 = (FloatingActionButton) findViewById(R.id.fab3);
        fab4 = (FloatingActionButton) findViewById(R.id.fab4);

        photoView = findViewById(R.id.photoView);
        photoView.setImageResource(R.drawable.camera);
        textView = findViewById(R.id.textView);

        fab.setOnClickListener(this);
        fab1.setOnClickListener(this);
        fab2.setOnClickListener(this);
        fab3.setOnClickListener(this);
        fab4.setOnClickListener(this);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if(requestCode == INTENT_REQUEST_ALBUM && resultCode == RESULT_OK) {
            try {
                Uri contentUri = data.getData();
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contentUri);
                photoView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (requestCode == INTENT_REQUEST_CAMERA && resultCode == RESULT_OK) {
            try {
                Log.v("fab1", "intent request");
                File f = new File(currentPhotoPath);
                Uri contentUri = Uri.fromFile(f);
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contentUri);
                photoView.setImageBitmap(bitmap);
            }catch(Exception e){
                e.printStackTrace();
            }
        }
    }

    public byte[] getBytes(InputStream is) throws IOException {
        ByteArrayOutputStream byteBuff = new ByteArrayOutputStream();
        
        int buffSize = 1024;
        byte[] buff = new byte[buffSize];
        
        int len = 0;
        while ((len = is.read(buff)) != -1) {
            byteBuff.write(buff, 0, len);
        }
        
        return byteBuff.toByteArray();
    }
    
    private void uploadImage(byte[] imageBytes) {
        Log.d("uploadImage","hi");
        OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.MINUTES)
            .readTimeout(90, TimeUnit.SECONDS)
            .writeTimeout(90, TimeUnit.SECONDS)
            .build();
        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build();
        
        RetrofitInterface retrofitInterface = retrofit.create(RetrofitInterface.class);
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/jpeg"), imageBytes);
    
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", "image.jpg", requestFile);
        Call<Response> call = retrofitInterface.uploadImage(body);
        call.enqueue(new Callback<Response>() {
            @Override
            public void onResponse(Call<Response> call, retrofit2.Response<Response> response) {
                if(response.isSuccessful()) {
                    Response responseBody = response.body();
                    imageString = responseBody.getImage();
                    celebrityString = responseBody.getCelebrity();
                    textView.setText(celebrityString);
                    textView.setVisibility(View.VISIBLE);
                    byte[] decodedString = null;
                    try {
                        decodedString = Base64.decode(imageString, Base64.DEFAULT);
                        System.out.println(decodedString.length);
                        Bitmap bitmapImage = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                        photoView.setImageBitmap(bitmapImage);
                    } catch (IllegalArgumentException e) {
                        photoView.setImageResource(R.drawable.xgiyhona_retry);
                    }
                    
//                    textView.setText(imageString);
                    Log.d("uploadImage","onResponse");
                } else {
                    ResponseBody errorBody = response.errorBody();
                    
                    Gson gson= new Gson();
                    
                    Log.d("uploadImage", "ERROR response");
    
                    try {
                        Response errorResponse = gson.fromJson(errorBody.string(), Response.class);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
    
            @Override
            public void onFailure(Call<Response> call, Throwable t) {
                Log.d("uploadImage", "FAILED");
                t.printStackTrace();
            }
        });
    }
    
    @Override
    public void onClick(View view) {
        if(view.getId() == R.id.fab2) {
            Log.d("fab2", "clicked");
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/");
            startActivityForResult(intent, INTENT_REQUEST_ALBUM);
            Log.d("fab2", "skipped");
        }
        if(view.getId() == R.id.fab1) {
            Log.d("fab1", "clicked");
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (intent.resolveActivity(this.getPackageManager()) != null) {
                File photoFile = null;
                try {
                    photoFile = createImageFile();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                if (photoFile != null) {
                    Uri providerURI = FileProvider.getUriForFile(this, this.getPackageName(), photoFile);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, providerURI);
                    startActivityForResult(intent, INTENT_REQUEST_CAMERA);
                }
            }
            Log.d("fab1", "skipped");
        }
        if(view.getId() == R.id.fab) {
            anim();
        }
        if(view.getId() == R.id.fab3) {
            bitmap = rotateImage(bitmap, 90);
            photoView.setImageBitmap(bitmap);
        }
        if(view.getId() == R.id.fab4){
            anim();
            uploadImage(bitmapToByteArray(bitmap));
            textView.setVisibility(View.INVISIBLE);
        }
    }

    private File createImageFile() throws IOException {
        String imgFileName = System.currentTimeMillis() + ".jpg";
        File imageFile;
        File storageDir = new File(Environment.getExternalStorageDirectory() + "/Pictures", "ireh");
        if(!storageDir.exists()){
            storageDir.mkdirs();
        }
        imageFile = new File(storageDir,imgFileName);
        currentPhotoPath = imageFile.getAbsolutePath();
        return imageFile;

    }

    public void anim() {

        if (isFabOpen) {
            fab1.startAnimation(fab_close);
            fab2.startAnimation(fab_close);
            fab3.startAnimation(fab_close);
            fab4.startAnimation(fab_close);
            fab1.setClickable(false);
            fab2.setClickable(false);
            fab3.setClickable(false);
            fab4.setClickable(false);
            isFabOpen = false;
        } else {
            fab1.startAnimation(fab_open);
            fab2.startAnimation(fab_open);
            fab3.startAnimation(fab_open);
            fab4.startAnimation(fab_open);
            fab1.setClickable(true);
            fab2.setClickable(true);
            fab3.setClickable(true);
            fab4.setClickable(true);
            isFabOpen = true;
        }
    }

    public Bitmap rotateImage(Bitmap src, float degree) {

        // Matrix 객체 생성
        Matrix matrix = new Matrix();
        // 회전 각도 셋팅
        matrix.postRotate(degree);
        // 이미지와 Matrix 를 셋팅해서 Bitmap 객체 생성
        return Bitmap.createBitmap(src, 0, 0, src.getWidth(),src.getHeight(), matrix, true);
    }

    public byte[] bitmapToByteArray(Bitmap $bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        $bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }
}
