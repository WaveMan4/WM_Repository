package kepnang.gilles.jewelmatcher;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.lifecycle.ViewModelProviders;

import android.view.View;


import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.ShapeDrawable;
import android.util.Log;
import android.view.MotionEvent;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


public class GameActivity extends AppCompatActivity {

    private Bundle recoveredBundle = null;

    //Objects for drawing jewels
    private DrawingArea drawingArea;
    private Drawable squareDrawable;
    private Drawable circleDrawable;
    private ShapeDrawable diamondDrawable;
    private ShapeDrawable gemStoneDrawable;
    private ShapeDrawable noShapeDrawable;
    private Drawable drawableToUse;
    private Paint diamondPaint;
    private Paint diamondStrokePaint;
    private Paint diamondFillPaint;
    private Paint gemStonePaint;
    private Paint gemStoneStrokePaint;
    private Paint gemStoneFillPaint;
    private int diamondFillColor;
    private int gemStoneFillColor;
    private int noShapeFillColor;
    private int scoreTotal = 0;

    //Variables for grid gameplay
    private short gameRow = 0;
    private short gameColumn = 0;
    private int jewelNumber;
    private int gameGrid[][] = null;
    private Random rand = new Random();
    private float jewelSize;
    private int lineColor;
    private float numberSize;
    private float letterSize;
    private float touchSlop;
    private Thing tappedJewel;

    //Variables for handling jewel data
    private List<Thing> things = new ArrayList<>();
    private float currentX;
    private float currentY;
    private ArrayList<Thing> thingsAtBounds;

    //flags for game modes
    private volatile boolean blink = false; //blinking/highlighting
    private volatile boolean match = false; //match flag
    private volatile boolean blank = false; //match flag
    private volatile boolean newjewels = false; //match flag
    private volatile boolean shuffled = false; //shuffle jewels

    //Global variables for grid matches
    ArrayList<int[]> horizontalMatches = new ArrayList<int[]>();
    ArrayList<int[]> verticalMatches  = new ArrayList<int[]>();
    ArrayList<int[]> allMatches  = new ArrayList<int[]>();


    /////////////////////////////////////////////////////////////////////////////////////////
    //                      RUNNABLE THREADS FOR GAME                                      //
    /////////////////////////////////////////////////////////////////////////////////////////
    //Runnable for blinks/highlights
    private Runnable blinker = new Runnable() {
        @Override
        public void run() {
            try {
                blink = true;
                drawingArea.postInvalidate();
                Thread.sleep(500);
                blink = false;
                drawingArea.postInvalidate();
                Thread.sleep(500);
                blink = true;
                drawingArea.postInvalidate();
                Thread.sleep(500);
                blink = false;
                drawingArea.postInvalidate();
                tappedJewel = null;
            } catch (InterruptedException e) {
                blink = false;
            }
        }
    };

    //Runnable for animation
    //  Animator sequence is as follows:
    //  --> 1) blink the matched items twice
    //  --> 2) blank out the matched items
    //  --> 3) insert new jewels into grid
    private Runnable animator = new Runnable() {
        @Override
        public void run() {
            try {
                //Refresh all matches
                drawingArea.createHorizontalMatches();
                drawingArea.createVerticalMatches();
                allMatches = drawingArea.outputAllMatches();

                if (allMatches.size() > 0) {
                    // 1) blink the matched items twice
                    match = true;
                    drawingArea.postInvalidate();
                    Thread.sleep(350);
                    match = false;
                    drawingArea.postInvalidate();
                    Thread.sleep(350);
                    match = true;
                    drawingArea.postInvalidate();
                    Thread.sleep(350);
                    match = false;
                    drawingArea.postInvalidate();
                    Thread.sleep(350);
                    match = true;
                    drawingArea.postInvalidate();
                    Thread.sleep(350);
                    match = false;
                    drawingArea.postInvalidate();

                    drawingArea.markTheMatches();
                    // 2) blank out the matched items
                    drawingArea.postInvalidate();
                    Thread.sleep(700);
                    drawingArea.postInvalidate();
                    drawingArea.prepareForNewJewels();

                    // 3) insert new jewels into grid
                    drawingArea.fillInNewJewels();
                    Thread.sleep(700);
                    drawingArea.postInvalidate();
                }

                //Refresh all matches
                drawingArea.createHorizontalMatches();
                drawingArea.createVerticalMatches();
                allMatches = drawingArea.outputAllMatches();
                if (allMatches.size() > 0) {
                    this.run();
                }

            } catch (InterruptedException e) {
                match = false;
            }
        }
    };

    //Runnable for blinks/highlights
    private Runnable shuffler = new Runnable() {
        @Override
        public void run() {
            try {
                drawingArea.shuffleGrid();
                Thread.sleep(250);
                drawingArea.postInvalidate();
            } catch (InterruptedException e) {
                Log.d("Interrupted Exception: ", e.getStackTrace().toString());
            }
        }
    };


    /**
     * LifeCycle Phase:
     */
    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        //outState.putAll(outState);
        //outState.putBundle("savedBundle", outState);
    }

    /**
     * LifeCycle Phase: onRestore
     */
    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        //scoreTotal = savedInstanceState.getInt("Score Total");
        //recoveredBundle = savedInstanceState.getBundle("savedBundle");
    }

    /**
     * LifeCycle Phase: onCreate
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ///////////////////////////////////////////////////////////////////////////////////////
        //                              INIT BOARD ELEMENTS                                  //
        //Create 8x8 GameGrid

        if (gameGrid == null) {
            gameGrid = new int[8][8];
            for (int row = 0; row < 8; row++) {
                for (int column = 0; column < 8; column++) {
                    jewelNumber = rand.nextInt(4); //Generate number between 0 - 3
                    gameGrid[row][column] = jewelNumber;
                    addThingToList(jewelNumber, column, row);
                }
            }
        }
        ///////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////
        //                               UI ELEMENTS FOR GAMES                                  //
        //////////////////////////////////////////////////////////////////////////////////////////
        //Set up the drawable
        drawableToUse = null;

        //Set Line attributes -ArrayList<Thing> Line is for connecting shapes
        lineColor = getResources().getColor(R.color.lineColor);
        touchSlop = getResources().getDimension(R.dimen.touch_slop);

        //Set Number attributes
        numberSize = getResources().getDimension(R.dimen.numberSize);
        letterSize = getResources().getDimension(R.dimen.letterSize);

        //Set jewel size as 40 dp
        jewelSize = getResources().getDimension(R.dimen.shapeSize);

        //Set colors for the Shapes
        diamondFillColor = getResources().getColor(R.color.diamondColor);
        gemStoneFillColor = getResources().getColor(R.color.gemStoneColor);
        noShapeFillColor = getResources().getColor(R.color.noShapeColor);

        //Draw 2 objects from XML
        squareDrawable = getResources().getDrawable(R.drawable.square);
        circleDrawable = getResources().getDrawable(R.drawable.circle);

        //Define stroke width for Jewels
        float strokeWidth = getResources().getDimension(R.dimen.strokeWidth);
        //Get statelist for grid highlights
        ColorStateList strokeColor = getResources().getColorStateList(R.color.stroke);

        //Draw 2 objects programmatically:
        //Create a Diamond "Drawable"
        diamondDrawable = createDiamond((int) strokeWidth, diamondFillColor, strokeColor);
        //Create a GemStone "Drawable"
        gemStoneDrawable = createGemStone((int) strokeWidth, gemStoneFillColor, strokeColor);
        //Create a GemStone "Drawable"
        noShapeDrawable = createNoShape((int) strokeWidth, noShapeFillColor, strokeColor);
        //////////////////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////
        //                                  DRAWING AREA                                        //
        //////////////////////////////////////////////////////////////////////////////////////////
        //Create custom view (aka Drawing Area)
        drawingArea = new DrawingArea(this);
        //Assign Custom View as the view for this Game Activity
        setContentView(drawingArea);
        new Thread(animator).start();
        //////////////////////////////////////////////////////////////////////////////////////////

    }

    ////////////////////////////////////////////////////////////////////////////////////
    //GRID MEMBER FUNCTIONS
    ////////////////////////
    //Add Thing object to list (as a new object)
    private void addThingToList(int number, int column, int row) {
        switch(number) {
            case 0:
                //add square
                things.add(new Thing(column, row, Thing.Type.Square,
                        thingBounds(column, row, (int) jewelSize)));
                break;
            case 1:
                //add circle
                things.add(new Thing(column, row, Thing.Type.Circle,
                        thingBounds(column, row, (int) jewelSize)));
                break;
            case 2:
                //add diamond
                things.add(new Thing(column, row, Thing.Type.Diamond,
                        thingBounds(column, row, (int) jewelSize)));
                break;
            case 3:
                //add gemstone
                things.add(new Thing(column, row, Thing.Type.GemStone,
                        thingBounds(column, row, (int) jewelSize)));
                break;
            case 5:
                //add gemstone
                things.add(new Thing(column, row, Thing.Type.NoShape,
                        thingBounds(column, row, (int) jewelSize)));
                break;

        }
    }
    //ThingBounds functions
    private Rect thingBounds(int x, int y, int size) {
        int halfSize = size / 2;
        return new Rect(x - halfSize,
                y - halfSize,
                x + halfSize,
                y + halfSize);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////

    //Create a separate instance of the diamond image (each time)
    private ShapeDrawable createDiamond(int strokeWidth, int diamondFillColor, ColorStateList strokeColor) {
        final Diamond diamond = new Diamond(strokeWidth, diamondFillColor, strokeColor);
        ShapeDrawable shapeDrawable = new ShapeDrawable(diamond) {

            //Observe state changes for each created Diamond
            @Override
            protected boolean onStateChange(int[] stateSet) {
                diamond.setState(stateSet);
                return super.onStateChange(stateSet);
            }
            //View class must pass state down to its Drawable
            //	by being "stateful", the Drawable can be moved while on canvas
            @Override
            public boolean isStateful() {
                return true;
            }

        };
        shapeDrawable.setIntrinsicHeight((int) jewelSize);
        shapeDrawable.setIntrinsicWidth((int) jewelSize);
        shapeDrawable.setBounds(0, 0, (int) jewelSize, (int) jewelSize);
        return shapeDrawable;
    }
    //Create a separate instance of the gemstone image (each time)
    private ShapeDrawable createGemStone(int strokeWidth, int gemStoneFillColor, ColorStateList strokeColor) {
        final GemStone gemStone = new GemStone(strokeWidth, gemStoneFillColor, strokeColor);
        ShapeDrawable shapeDrawable = new ShapeDrawable(gemStone) {

            //Observe state changes for each created Gemstone
            @Override
            protected boolean onStateChange(int[] stateSet) {
                gemStone.setState(stateSet);
                return super.onStateChange(stateSet);
            }
            //View class must pass state down to its Drawable
            //	by being "stateful", the Drawable can be moved while on canvas
            @Override
            public boolean isStateful() {
                return true;
            }

        };
        shapeDrawable.setIntrinsicHeight((int) jewelSize);
        shapeDrawable.setIntrinsicWidth((int) jewelSize);
        shapeDrawable.setBounds(0, 0, (int) jewelSize, (int) jewelSize);
        return shapeDrawable;
    }

    //Create a separate instance of the gemstone image (each time)
    private ShapeDrawable createNoShape(int strokeWidth, int gemStoneFillColor, ColorStateList strokeColor) {
        final NoShape noShape = new NoShape(strokeWidth, gemStoneFillColor, strokeColor);
        ShapeDrawable shapeDrawable = new ShapeDrawable(noShape) {

            //Observe state changes for each created Gemstone
            @Override
            protected boolean onStateChange(int[] stateSet) {
                noShape.setState(stateSet);
                return super.onStateChange(stateSet);
            }
            //View class must pass state down to its Drawable
            //	by being "stateful", the Drawable can be moved while on canvas
            @Override
            public boolean isStateful() {
                return true;
            }

        };
        shapeDrawable.setIntrinsicHeight((int) jewelSize);
        shapeDrawable.setIntrinsicWidth((int) jewelSize);
        shapeDrawable.setBounds(0, 0, (int) jewelSize, (int) jewelSize);
        return shapeDrawable;
    }
    //private inner class - to add Shapes to the View
    private class DrawingArea extends View {

        //////////////////////////////////////////////////////////////////////////////////////////
        //                                  GAME ELEMENTS                                       //
        //////////////////////////////////////////////////////////////////////////////////////////

        //paint variable for number
        private Paint scorePaint = new Paint();
        private Paint statusPaint = new Paint();
        //thing variable for selected jewel
        private Thing selectedJewel = null;

        //////////////////////////////////////////////////////////////////////////////////////////
        //                                    GAME MODES                                        //
        //////////////////////////////////////////////////////////////////////////////////////////
        private boolean gameBaseState = true;
        private boolean movingJewel = false;
        private boolean gamePaused = false;
        private boolean swapElementsWithinMatches = false;
        private boolean jewelIsTapped = false;
        //////////////////////////////////////////////////////////////////////////////////////////
        //                               GAME GRID VARIABLES                                    //
        //////////////////////////////////////////////////////////////////////////////////////////
        private int jewelsAtThisCoordinate = 0;
        private Thread animatorThread = null;

        //////////////////////////////////////////////////////////////////////////////////////////
        //                                 DISPLAY MODES                                        //
        //////////////////////////////////////////////////////////////////////////////////////////
        private boolean isLandscape = false;

        /**********************\
         * Create Drawing Area *
         \**********************/
        public DrawingArea(Context context) {
            super(context);
            scorePaint.setColor(lineColor);
            scorePaint.setTextSize(numberSize);
            statusPaint.setColor(lineColor);
            statusPaint.setTextSize(letterSize);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        //                              HANDLE TOUCH EVENTS                                    //
        /////////////////////////////////////////////////////////////////////////////////////////
        @Override
        public boolean onTouchEvent(MotionEvent event) {
            //Nested switch-case logic:
            //	Notes:
            // 	for each Motion event,

            switch(event.getAction()) {
                case MotionEvent.ACTION_DOWN:

                    //establish objects currently at the bounds
                    findThingsAtBounds();
                    //first get coordinates (x,y) of button press
                    int ccordinateX = (int) event.getX();
                    int ccordinateY = (int) event.getY();
                    //then compare each coordinate to things at the bounds
                    if (!gamePaused) {
                        for (Thing thing : thingsAtBounds) {
                            if (ccordinateX < thing.getBounds().left &&
                                    thing.getColumn() == 0) {
                                gamePaused = true;
                                break;
                            } else if (ccordinateY < thing.getBounds().top &&
                                    thing.getRow() == 0) {
                                gamePaused = true;
                                break;
                            } else if (ccordinateX > thing.getBounds().right &&
                                    thing.getColumn() == 7) {
                                gamePaused = true;
                                break;
                            } else if (ccordinateY > thing.getBounds().bottom &&
                                    thing.getRow() == 7) {
                                gamePaused = true;
                                break;
                            } else {
                            }
                        }
                    } else gamePaused = false;

                    //////////////////////////////////////////////////////////////////////////////
                    //Now that we have checked for pause...
                    //we can proceed with further game states (for an un-paused game
                    //Enter game in  gameBaseState state
                    //-->Transition into default gameBaseState
                    if (gameBaseState && !gamePaused) {
                        gameBaseState = false;
                        //get x,y coordinates
                        //find jewels by coordinates
                        selectedJewel = locateJewel((int) event.getX(), (int) event.getY());


                        //note: when you click in open space it returns null for a selected thing
                        // handle blank clicks by checking whether it is null or not
                        if(selectedJewel != null) {
                            tappedJewel = selectedJewel;
                            things.remove(selectedJewel); //removes item from current position
                            things.add(selectedJewel); //re=adds item on the front
                            new Thread(blinker).start();
                        }
                        return true;
                    }
                    invalidate();
                    return true;
                case MotionEvent.ACTION_MOVE:

                    //check current position
                    currentX = event.getX();
                    currentY = event.getY();

                    if (selectedJewel != null) {
                        float jewelX = selectedJewel.getBounds().left;
                        float jewelY = selectedJewel.getBounds().top;

                        float diffX = Math.abs(currentX - jewelX);
                        float diffY = Math.abs(currentY - jewelY);

                        //compare current position to touch slop
                        if (diffX > touchSlop && diffY > touchSlop) {
                            movingJewel = true;
                            for (Thing thing : thingsAtBounds) {
                                if (selectedJewel.getBounds().left < thing.getBounds().left &&
                                        thing.getColumn() == 0) {
                                    shuffled = true;
                                    break;
                                } else if (selectedJewel.getBounds().top < thing.getBounds().top &&
                                        thing.getRow() == 0) {
                                    shuffled = true;
                                    break;
                                } else if (selectedJewel.getBounds().right > thing.getBounds().right &&
                                        thing.getColumn() == 7) {
                                    shuffled = true;
                                    break;
                                } else if (selectedJewel.getBounds().bottom > thing.getBounds().bottom &&
                                        thing.getRow() == 7) {
                                    shuffled = true;
                                    break;
                                } else {
                                }
                            }
                            if (!shuffled) {
                                //if current position is greater than touch slop, then set item to new position
                                selectedJewel.setBounds(thingBounds((int) event.getX(),
                                        (int) event.getY(),
                                        (int) jewelSize));
                            }
                        }
                        return true;
                    }
                    invalidate();
                    return true;
                case MotionEvent.ACTION_UP: //Action-Up indicates "potential swap"

                    //check if current coordinates have more than one jewel
                    int x = (int) event.getX();
                    int y = (int) event.getY();
                    jewelsAtThisCoordinate = numberofJewelsAt(x, y);


                    if (shuffled && jewelsAtThisCoordinate != 2) {
                        new Thread(shuffler).start();
                        scoreTotal -= 10;
                        shuffled = false;
                    }
                    else if (jewelsAtThisCoordinate == 2 && !shuffled) {
                        //retrieve jewels
                        ArrayList<Thing> retrievedJewels = retrieveJewelsAt(x, y);
                        Thing thing1 = retrievedJewels.get(0);
                        Thing thing2 = retrievedJewels.get(1);

                        //Check if the two jewels are adjacent
                        //-->If so, then proceed to predict that a swap will cause a match
                        if (jewelsAreAdjacent(thing1.getColumn(),
                                thing1.getRow(),
                                thing2.getColumn(),
                                thing2.getRow())) {

                            //check if swap can create three or more in a row
                            int thing1Element =
                                    gameGrid[thing1.getRow()][thing1.getColumn()];
                            int thing2Element =
                                    gameGrid[thing2.getRow()][thing2.getColumn()];

                            //preliminary swap (within the grid)
                            gameGrid[thing1.getRow()][thing1.getColumn()] = thing2Element;
                            gameGrid[thing2.getRow()][thing2.getColumn()] = thing1Element;

                            //Find horizontal matches and vertical matches
                            horizontalMatches =
                                    horizontalScan(gameGrid);
                            verticalMatches =
                                    verticalScan(gameGrid);
                            allMatches.clear();
                            allMatches.addAll(horizontalMatches);
                            allMatches.addAll(verticalMatches);

                            //determine that swapped items are within the matches
                            swapElementsWithinMatches = false;
                            for (int[] intpointer : allMatches) {
                                if ((intpointer[0] == thing1.getRow() &&
                                        intpointer[1] == thing1.getColumn()) ||
                                        (intpointer[0] == thing2.getRow() &&
                                                intpointer[1] == thing2.getColumn())) {
                                    swapElementsWithinMatches = true;
                                    break;
                                }
                            }
                            //if the arraylists have any matches,
                            // and the swapped elements are part of the matches
                            if (((horizontalMatches.size() + verticalMatches.size()) > 0) &&
                                    (swapElementsWithinMatches)) {
                                //remove previous jewels
                                things.remove(thing1);
                                things.remove(thing2);
                                //add new swapped jewels to list
                                addThingToList(thing2Element,
                                        thing1.getColumn(),
                                        thing1.getRow());
                                addThingToList(thing1Element,
                                        thing2.getColumn(),
                                        thing2.getRow());
                                selectedJewel = null;
                                new Thread(animator).start();
                            }
                            //ELSE SWAP BACK
                            else {
                                //At this point:
                                //  thing1 is in the place of thing2, and
                                //  thing2 is in the place of thing1
                                thing1Element =
                                        gameGrid[thing1.getRow()][thing1.getColumn()];
                                thing2Element =
                                        gameGrid[thing2.getRow()][thing2.getColumn()];
                                //swap back the jewels
                                gameGrid[thing1.getRow()][thing1.getColumn()] = thing2Element;
                                gameGrid[thing2.getRow()][thing2.getColumn()] = thing1Element;
                            }
                        }
                    }

                    movingJewel = false;
                    gameBaseState = true;
                    selectedJewel = null;
                    invalidate();
                    return true;

            }
            return super.onTouchEvent(event);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////////////
        //MEMBER FUNCTIONS
        //////////////////
        //find thing within list
        private Thing findThingAt(int column, int row) {
            //locate lastmost (recent) thing on the top
            for(int i = things.size() - 1; i >= 0; i--) {
                Thing thing = things.get(i);
                if (thing.getColumn() == column && thing.getRow() == row) {
                    return thing;
                }
            }
            return null;
        }
        //find things at the bounds (within the thing-list)
        private void findThingsAtBounds() {
            thingsAtBounds = new ArrayList<Thing>();
            for (int row = 0; row < 8; row++)
                for (int column = 0; column < 8; column++)
                    if (row == 0 || row == 7 ||
                            column == 0 || column == 7)
                        thingsAtBounds.add(findThingAt(column, row));
        }
        //locate Jewel
        private Thing locateJewel(int x, int y) {
            //locate lastmost (recent) thing on the top
            for(int i = things.size() - 1; i >= 0; i--) {
                Thing thing = things.get(i);
                if (thing.getBounds().contains(x, y)) {
                    return thing;
                }
            }
            return null;
        }
        //Read number of jewels within list at this location
        private int numberofJewelsAt(int x, int y) {
            int numberofJewels = 0;
            for(int i = things.size() - 1; i >= 0; i--) {
                Thing thing = things.get(i);
                if (thing.getBounds().contains(x, y)) {
                    numberofJewels++;
                }
            }
            return numberofJewels;
        }
        //Read number of jewels within list at this location
        private ArrayList<Thing> retrieveJewelsAt(int x, int y) {
            ArrayList<Thing> listofJewels = new ArrayList<Thing>();
            for(int i = things.size() - 1; i >= 0; i--) {
                Thing thing = things.get(i);
                if (thing.getBounds().contains(x, y)) {
                    listofJewels.add(thing);
                }
            }
            return listofJewels;
        }
        //determine if jewels in positions are adjacent
        private boolean jewelsAreAdjacent(int x1, int y1, int x2, int y2) {

            int diffx = x1 - x2;
            int diffy = y2 - y1;

            boolean adjacentx = (diffx == 1 || diffx == -1);
            boolean adjacenty = (diffy == 1 || diffy == -1);
            boolean samex = (x1 == x2);
            boolean samey = (y1 == y2);

            if (adjacentx & samey) return true;
            if (adjacenty & samex) return true;

            //default case return false
            return false;
        }

        //Vertical Scan
        //-->receives a copy of gameGrid for local use
        ArrayList <int[]> verticalScan(int inputGrid[][]) {
            ArrayList <int[]> jewelsInAColumn =
                    new ArrayList <int[]>();
            int currentJewel;
            int verticalConsecutiveMatches = 0;
            int row = 0;
            int column = 0;
            int newCoordinates[] = null;
            ArrayList<int[]> matchesInAColumn = new ArrayList<>();

            for (column = 0; column < 8; column++)
            {
                //while loop has not reached the 8th element
                while (row < 8) {

                    //From Beginning to End:

                    //For first jewel in the row:
                    //  1) establish first jewel,
                    //  2) establish (x,y) coordinates
                    //  3) increment vertical matches
                    if (row == 0) {

                        newCoordinates = new int[2];

                        newCoordinates[0] = row;
                        newCoordinates[1] = column;

                        verticalConsecutiveMatches++;
                        matchesInAColumn.add(newCoordinates);

                    }
                    //From second jewel in the row, until the last
                    //first compare current to previous
                    if (row >= 1) {

                        //Assign as current Jewel
                        currentJewel = inputGrid[row][column];

                        //increment horizontal-matches ... until the current is different from previous
                        if (currentJewel == inputGrid[row-1][column]) {

                            newCoordinates = new int[2];

                            newCoordinates[0] = row;
                            newCoordinates[1] = column;

                            verticalConsecutiveMatches++;
                            matchesInAColumn.add(newCoordinates);

                            //handle end case when row = 7
                            if (verticalConsecutiveMatches >= 3 && row == 7) {
                                //add List to Hashmap jewelsInARow
                                jewelsInAColumn.addAll(matchesInAColumn);
                            }

                        }
                        else {
                            //next check if horizontal-matches >= 3
                            if (verticalConsecutiveMatches >= 3) {
                                //add List to Hashmap jewelsInARow
                                jewelsInAColumn.addAll(matchesInAColumn);
                            }
                            //set horizontal-matches to 1 for a new element
                            verticalConsecutiveMatches = 1;
                            //clear List and add current jewel
                            matchesInAColumn.clear();

                            newCoordinates = new int[2];

                            newCoordinates[0] = row;
                            newCoordinates[1] = column;
                            matchesInAColumn.add(newCoordinates);
                        }
                    }
                    row++;
                }
                row = 0;
                //reset vertical-matches to 0
                verticalConsecutiveMatches = 0;
                //clear List and add current jewel
                matchesInAColumn.clear();
            }
            return jewelsInAColumn;
        }
        //Horizontal Scan
        //-->receives a copy of gameGrid for local use
        ArrayList <int[]> horizontalScan(int inputGrid[][]) {
            ArrayList <int[]> jewelsInARow =
                    new ArrayList <int[]>();
            int currentJewel;
            int horizontalConsecutiveMatches = 0;
            int row = 0;
            int column = 0;
            int newCoordinates[] = null;
            ArrayList <int[]> matchesInARow = new ArrayList<>();

            for (row = 0; row < 8; row++)
            {
                //while loop has not reached the 8th element
                while (column < 8) {

                    //From Beginning to End:

                    //For first jewel in the row:
                    //  1) establish first jewel,
                    //  2) establish (x,y) coordinates
                    //  3) increment horizontal matches
                    if (column == 0) {

                        newCoordinates = new int[2];

                        newCoordinates[0] = row;
                        newCoordinates[1] = column;

                        horizontalConsecutiveMatches++;
                        matchesInARow.add(newCoordinates);

                    }
                    //From second jewel in the row, until the last
                    //first compare current to previous
                    if (column >= 1) {

                        //Assign as current Jewel
                        currentJewel = inputGrid[row][column];

                        //increment horizontal-matches ... until the current is different from previous
                        if (currentJewel == inputGrid[row][column-1]) {

                            newCoordinates = new int[2];

                            newCoordinates[0] = row;
                            newCoordinates[1] = column;

                            horizontalConsecutiveMatches++;
                            matchesInARow.add(newCoordinates);

                            //handle end case when column = 7
                            if (horizontalConsecutiveMatches >= 3 && column == 7) {
                                //add List to Hashmap jewelsInARow
                                jewelsInARow.addAll(matchesInARow);
                            }

                        }
                        else {
                            //next check if horizontal-matches >= 3
                            if (horizontalConsecutiveMatches >= 3) {
                                //add List to Hashmap jewelsInARow
                                jewelsInARow.addAll(matchesInARow);
                            }
                            //set horizontal-matches to 1 for a new element
                            horizontalConsecutiveMatches = 1;
                            //clear List and add current jewel
                            matchesInARow.clear();

                            newCoordinates = new int[2];

                            newCoordinates[0] = row;
                            newCoordinates[1] = column;
                            matchesInARow.add(newCoordinates);
                        }
                    }
                    column++;
                }
                column = 0;
                //reset horizontal-matches to 0
                horizontalConsecutiveMatches = 0;
                //clear List and add current jewel
                matchesInARow.clear();
            }
            return jewelsInARow;
        }

        //Shuffle grid
        private void shuffleGrid() {
            Random listRand = new Random();
            List <Thing> substituteThings = new ArrayList<>();
            Thing oldThing = null;
            Thing newThing = null;
            int currentJewel = 5;   //five is a default value (empty space)

            for (int row = 0; row < 8; row++) {
                for (int column = 0; column < 8; column++) {
                    //transfer thing from "things" to "substituteThings"
                    oldThing = things.remove(listRand.nextInt(things.size()));

                    switch (oldThing.getType()) {
                        case Square:
                            gameGrid[row][column] = 0;
                            break;
                        case Circle:
                            gameGrid[row][column] = 1;
                            break;
                        case Diamond:
                            gameGrid[row][column] = 2;
                            break;
                        case GemStone:
                            gameGrid[row][column] = 3;
                            break;
                        case NoShape:
                            gameGrid[row][column] = 5;
                            break;
                    }
                    oldThing.setColumn(column);
                    oldThing.setRow(row);
                    substituteThings.add(oldThing);
                }
            }
            things.addAll(substituteThings);
        }

        //find thing within list
        private boolean coodinateIsMatched(int row, int column) {
            //locate lastmost (recent) thing on the top
            for(int intcouple[] : allMatches) {
                if (intcouple[0] == row && intcouple[1] == column)
                {
                    return true;
                }
            }
            return false;
        }

        //////////////////////////////////////////////////////////////////////////////////////
        //FUNCTIONS FOR ADJUSTING CELLS AFTER A MATCH
        //  1) Mark the matched cordinates as arbitrary number '5'
        //  2) Push down all unmatched jewels, push up the marked '5' jewels
        //  3) Generate new jewels
        //////////////////////////////////////////////////////////////////////////////////////
        private void createHorizontalMatches() {
            horizontalMatches.clear();
            horizontalMatches = horizontalScan(gameGrid);
        }
        private void createVerticalMatches() {
            verticalMatches.clear();
            verticalMatches = verticalScan(gameGrid);
        }
        private ArrayList<int[]> outputAllMatches() {
            allMatches.clear();
            allMatches.addAll(horizontalMatches);
            allMatches.addAll(verticalMatches);
            return allMatches;
        }

        private void markTheMatches() {
            int retrievedMatch[] = new int[2];
            //First pass: mark the matched cordinates as arbitrary number '5'
            for (int row = 0; row < 8; row++) {
                for (int column = 0; column < 8; column++) {
                    //check if current (row, column) coordinates equal to "matchedcoordinates"
                    //if equal, then do the following steps below:
                    if (coodinateIsMatched(row,column)) {
                        //find current thing
                        Thing thing = findThingAt(column, row);

                        //mark that specific cell as '5'
                        gameGrid[row][column] = 5;

                        //add new thing; remove thing
                        addThingToList(gameGrid[row][column], column, row);
                        things.remove(thing);

                        //increment score
                        scoreTotal++;
                    }
                }
            }
            //Final Step:
            //Re-ascribe a new Thing object for each "jewel Number" in the grid
            //ascribeNewThings();
        }
        private void prepareForNewJewels() {
            int retrievedMatch[] = new int[2];
            //Second pass: push down all values, so they are below arbitrary '5'
            int substitute = 0;
            for (int column = 0; column < 8; column++) {
                for (int row = 0; row < 8; row++) {
                    if (gameGrid[row][column] == 5) //if current value is '5' and not on the top
                    {
                        while (row != 0) {
                            Thing thing1 = findThingAt(column, row-1);
                            Thing thing2 = findThingAt(column, row);

                            substitute = gameGrid[row-1][column];
                            gameGrid[row-1][column] = gameGrid[row][column];
                            gameGrid[row][column] = substitute;
                            thing1.setRow(row);
                            thing2.setRow(row-1);
                            row--;
                        }
                        while (gameGrid[row][column] == 5)
                            row++;
                    }
                }
            }
            //Final Step:
            //Re-ascribe a new Thing object for each "jewel Number" in the grid
            //ascribeNewThings();
        }
        private void fillInNewJewels() {
            int retrievedMatch[] = new int[2];
            //Third pass: generate new ones
            for (int row = 0; row < 8; row++) {
                for (int column = 0; column < 8; column++) {
                    //check if array value == 5
                    if (gameGrid[row][column] == 5)	{
                        //find current thing
                        Thing thing = findThingAt(column, row);

                        //generate new random number
                        jewelNumber = rand.nextInt(4); //Generate number between 0 - 3

                        //add random number and shape to thing-list
                        gameGrid[row][column] = jewelNumber;

                        //add new thing; remove previous thing
                        addThingToList(gameGrid[row][column], column, row);
                        things.remove(thing);

                    }
                }
            }
            //Final Step:
            //check for next horizontal and vertical matches
            horizontalMatches =
                    horizontalScan(gameGrid);
            verticalMatches =
                    verticalScan(gameGrid);
            allMatches.clear();
            allMatches.addAll(horizontalMatches);
            allMatches.addAll(verticalMatches);
        }
        /////////////////////////////////////////////////////////////////////////////////////////

        //onDraw method:
        //	1) Draw Scoreboard
        //	2) Draw in Jewels
        //  3) Draw Grid Borders for Cells over the Jewels
        @Override
        protected void onDraw(Canvas canvas) {

            //set background color
            canvas.drawColor(Color.WHITE);

            if (getWidth() > getHeight()) {
                isLandscape = true;
            }
            else {
                isLandscape = false;
            }

            //Establish gridline dividers (tx, ty)
            int tx = isLandscape ? getWidth()/18 : getWidth() / 10;
            int ty = isLandscape ? getHeight()/10 : getHeight() / 14;

            if (isLandscape) {
                //Draw Scores
                canvas.drawText(Integer.toString(scoreTotal),
                        32,
                        96,
                        scorePaint);
            }
            else {
                //Draw Scores
                canvas.drawText(Integer.toString(scoreTotal),
                        32,
                        96,
                        scorePaint);
            }

            if (gamePaused)
            {
                if (isLandscape) {
                    //Draw Status
                    canvas.drawText("Game Paused (tap center of screen to unlock)",
                            64,
                            getHeight() - (ty) - 32,
                            statusPaint);
                }
                else {
                    //Draw Status
                    canvas.drawText("Game Paused (tap center of screen to unlock)",
                            64,
                            getHeight() - (2 * ty),
                            statusPaint);
                }
                //Draw "Pause" Status
                canvas.drawText("",
                        32,
                        getHeight() - (2 * ty),
                        statusPaint);
            }


            if (!gamePaused) {

                try {
                    //Draw Grid - gridlines with grid elements
                    for (Thing thing : things) {

                        //determine Jewel-to-draw, based on "jewel number"
                        switch (thing.getType()) {
                            case Square:
                                drawableToUse = squareDrawable;
                                break;
                            case Circle:
                                drawableToUse = circleDrawable;
                                break;
                            case Diamond:
                                drawableToUse = diamondDrawable;
                                break;
                            case GemStone:
                                drawableToUse = gemStoneDrawable;
                                break;
                            case NoShape:
                                drawableToUse = noShapeDrawable;
                                break;
                        }

                        if (gameBaseState) {
                            //2) Draw things at their positions, on top of the lines
                            if (isLandscape) {
                                //set thing to new location
                                thing.setBounds(new Rect(tx * (thing.getColumn() + 5) + 8,
                                                        (ty * (thing.getRow() + 1)) + 8,
                                                       (tx * (thing.getColumn() + 5 + 1)) - 8,
                                                     (ty * (thing.getRow() + 1 + 1)) - 8));
                            }
                            else {
                                //set thing to new location
                                thing.setBounds(new Rect(tx * (thing.getColumn() + 1) + 8,
                                                        (ty * (thing.getRow() + 2)) + 8,
                                                       (tx * (thing.getColumn() + 2)) - 8,
                                                     (ty * (thing.getRow() + 2 + 1)) - 8));
                            }
                        }

                        //Condition for tapped jewel
                        if (    (match && (coodinateIsMatched(thing.getRow(),
                                                              thing.getColumn()))) ||

                                (thing == selectedJewel ||
                                (blink &&
                                 tappedJewel != null &&
                                 thing.getType() == tappedJewel.getType()))
                        ) {
                            drawableToUse.setState(selectedState);
                        }
                        else {
                            drawableToUse.setState(unselectedState);
                        }

                        //dynamically draw the jewels jewels
                        drawableToUse.setBounds(thing.getBounds());
                        drawableToUse.draw(canvas);
                    }
                } catch (ConcurrentModificationException cme) {
                    invalidate();
                }
            }


        }
    }

    private static final int[] selectedState = {android.R.attr.state_selected};
    private static final int[] unselectedState = {};
}
