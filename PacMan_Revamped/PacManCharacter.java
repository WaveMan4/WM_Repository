///////////////////////////////////////////////////////////
//////// CLASS: PACMAN CHARACTER///////////////////////////
///////////////////////////////////////////////////////////
//Name: Gilles Kepnang

public class PacManCharacter 
{
    private int posX;
    private int posY;
    private String direction;
    
    public int cookiesEaten;
    public int spacesVisited;
    
    public PacManCharacter()
    {
        posX = 0;
        posY = 0;
        direction = ">";
        cookiesEaten = 0;
        spacesVisited = 0;
    }

    public int getPosX() 
    {
        return posX;
    }

    public void setPosX(int posX) 
    {
        this.posX = posX;
    }

    public int getPosY() 
    {
        return posY;
    }

    public void setPosY(int posY) 
    {
        this.posY = posY;
    }

    public String getDirection() 
    {
        return direction;
    }

    public void setDirection(String direction) 
    {
        this.direction = direction;
    }
    
    public void updatePacManPosition(int x, int y)
    {
        this.posX = x;
        this.posY = y;
    }
}