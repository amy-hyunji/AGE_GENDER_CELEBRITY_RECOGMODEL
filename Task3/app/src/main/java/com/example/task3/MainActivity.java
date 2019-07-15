package com.example.task3;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore.Audio.Media;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import com.google.gson.Gson;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import org.w3c.dom.Text;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import com.example.task3.Response;

public class MainActivity extends AppCompatActivity implements OnClickListener {
    
    public static final String URL = "http://143.248.36.213:3355/";
    
    private static final int INTENT_REQUEST_ALBUM = 100;
    
    private Button button;
    private ImageView imageView;
    private TextView textView;
    private String imageString;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        button = (Button) findViewById(R.id.button);
        imageView = (ImageView) findViewById(R.id.imageView);
//        textView = (TextView) findViewById(R.id.textView);
        
        button.setOnClickListener(this);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if(requestCode == INTENT_REQUEST_ALBUM && resultCode == RESULT_OK) {
            try {
                InputStream is = getContentResolver().openInputStream(data.getData());
                Log.d("onActivityResult", "here");
                uploadImage(getBytes(is));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
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
        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(URL)
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
                    byte[] decodedString = null;
                    try {
                        decodedString = Base64.decode(imageString, Base64.DEFAULT);
                        Bitmap bitmapImage = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                        imageView.setImageBitmap(bitmapImage);
                    } catch (IllegalArgumentException e) {
                        imageView.setImageResource(R.drawable.xgiyhona_retry);
                    }
                    
//                    textView.setText(imageString);
                    Log.d("uploadImage","onResponse");
                } else {
                    ResponseBody errorBody = response.errorBody();
                    
                    Gson gson= new Gson();
    
                    try {
                        Response errorResponse = gson.fromJson(errorBody.string(), Response.class);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
    
            @Override
            public void onFailure(Call<Response> call, Throwable t) {
        
            }
        });
    }
    
    @Override
    public void onClick(View view) {
        if(view.getId() == R.id.button) {
            Log.d("button", "clicked");
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/");
            startActivityForResult(intent, INTENT_REQUEST_ALBUM);
            Log.d("button", "skipped");
        }
    }
}
