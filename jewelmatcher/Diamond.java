package kepnang.gilles.jewelmatcher;

import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.shapes.Shape;

public class Diamond extends Shape {
    private int strokeWidth;
    private final int fillColor;
    private ColorStateList strokeColor;
    private Path path;
    private Paint strokePaint;
    private Paint fillPaint;

    //Constructor with input parameters
    public Diamond(int strokeWidth, int fillColor, ColorStateList strokeColor) {
        this.strokeWidth = strokeWidth;
        this.fillColor = fillColor;
        this.strokeColor = strokeColor;

        this.strokePaint = new Paint();
        this.strokePaint.setStyle(Paint.Style.STROKE);
        this.strokePaint.setColor(strokeColor.getColorForState(new int[0], 0));
        this.strokePaint.setStrokeJoin(Paint.Join.ROUND);
        this.strokePaint.setStrokeWidth(strokeWidth);

        this.fillPaint = new Paint();
        this.fillPaint.setStyle(Paint.Style.FILL);
        this.fillPaint.setColor(fillColor);
    }


    public void setState(int[] stateList) {
        this.strokePaint.setColor(strokeColor.getColorForState(stateList, 0));
    }

    @Override
    public void draw(Canvas canvas, Paint paint) {
        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, strokePaint);
    }

    //Define path for drawing diamond
    @Override
    protected void onResize(float width, float height) {
        super.onResize(width, height);
        path = new Path();

        //Draw a diamond
        path.moveTo(width/3, 0);
        path.lineTo((2 * width)/3, 0);
        path.lineTo(width, height/2);
        path.lineTo(width/2, height);
        path.lineTo(0, height/2);
        path.close(); //close the shape
    }
}
