#include <Servo.h>

Servo s0;
Servo s1;
Servo s2;
Servo s3;
Servo s4;

void setup() {
  Serial.begin(9600);
   
   s0.attach(3);//10;//pinky 0,a
   s1.attach(5);//9//ring 1,b
   s2.attach(6);//middle 2,c
   s3.attach(9);//5//index 3,d
   s4.attach(10);//3;//THUMB 4,e

   s0.write(0);
   s1.write(0);
   s2.write(0);
   s3.write(0);
   s4.write(0);
 
 }

  void loop() {
    
  if(Serial.available())
  {      s0.write(0);
         s1.write(0);
         s2.write(0);
         s3.write(0);
         s4.write(0);
      char x = Serial.read();
      Serial.println(x);
      
      if(x == 's')
      { 
        int arr[5] = {0,0,0,0,0};
        while( x!= 'z')
        { 
          x = Serial.read();
          Serial.println(x);  
            if(x=='a'){
            arr[0]=180;
            }
  
            if(x=='b'){
            arr[1]=150;
            }
            
            if(x=='c'){
            arr[2]=130;
            }
            
            if(x=='d'){
            arr[3]=180;
            }
            
            if(x=='e'){
            arr[4]=180;
            }
          
          }

            while( x != 'r')
            { x = Serial.read();
              Serial.println(x); 
            
            s0.write(arr[0]);
            s1.write(arr[1]);
            s2.write(arr[2]);
            s3.write(arr[3]);
            s4.write(arr[4]);
            delay(200);
            }
           
          
        }
      }
  }


