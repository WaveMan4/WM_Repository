package kepnang.gilles.jewelmatcher;

import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.shapes.Shape;

import android.content.res.ColorStateList;
        import android.graphics.Canvas;
        import android.graphics.Paint;
        import android.graphics.Path;
        import android.graphics.drawable.shapes.Shape;

public class GemStone extends Shape {
    private int strokeWidth;
    private final int fillColor;
    private ColorStateList strokeColor;
    private Path path;
    private Paint strokePaint;
    private Paint fillPaint;

    //Constructor with input parameters
    public GemStone(int strokeWidth, int fillColor, ColorStateList strokeColor) {
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

    //Define path for drawing gemstone
    @Override
    protected void onResize(float width, float height) {
        super.onResize(width, height);
        path = new Path();

        //Draw a gemstone
        path.moveTo(0, height/2);
        path.lineTo((3 * width)/5, 0);
        path.lineTo((3*width)/4, 0);
        path.lineTo(width, height/4);
        path.lineTo(width, height/2);
        path.lineTo((2*width)/5, height);
        path.lineTo(width/4, height);
        path.lineTo(0, (3*height)/4);
        path.close(); //close the shape
    }
}
