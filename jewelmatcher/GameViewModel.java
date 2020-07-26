package kepnang.gilles.jewelmatcher;

import android.app.Application;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class GameViewModel extends AndroidViewModel {

    public GameViewModel(@NonNull Application application) {
        super(application);
    }

    public MutableLiveData<ArrayList<int[]>> horizontalMatches = new MutableLiveData<>();
    public MutableLiveData<ArrayList<int[]>> verticalMatches = new MutableLiveData<>();
    public MutableLiveData<ArrayList<int[]>> allMatches = new MutableLiveData<>();
    public MutableLiveData<int[][]> gameGrid = new MutableLiveData<>();
    public MutableLiveData<List<Thing>> thingsToDisplay = new MutableLiveData<>();

}
