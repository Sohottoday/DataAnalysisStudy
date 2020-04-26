# 자바의 정석 예제 곱씹어보기

## 5. 배열

### 배열(Array)

- 예제 5-1

```java
class ArrayEx1 {
    public static void main(String[] args) {
        int[] score = new int[5];
        int k = 1;
        
        score[0] = 50;
        score[1] = 60;
        score[k+1] = 70;
        score[3] = 80;
        score[4] = 90;
        
        int tmp = score[k+2] 9 score[4];
                       
       for(int i=0 ; i<5; i++) {
               System.out.printf("score[%d]:%d%n", i, score[i]);
       }
        System.out.printf("tmp:%d%n", tmp);
    }
}
```

- 실행 결과

  score[0]:50

  score[1]:60

  score[2]:70

  score[3]:80

  score[4]:90

  tmp:170

---

- 예제 5-2

```java
import java.util.*;

class ArrayEx2 {
    public static void main(String[] args) {
        int[] iArr1 = new int[10];
        int[] iArr2 = new int[10];
        int[] iArr3 = new int[]{100, 95, 80, 70, 60};
        int[] iArr3 = {100, 95, 80, 70, 60};
        char[] chArr = {'a', 'b', 'c', 'd'};
        
        for(int i=0; i<iArr1.length ; i++) {
            iArr[i] = i+1;
        }
        
        for(int i=0; i<iArr2.length ; i++) {
            iArr2[i] = (int)(Math.random() * 10)+1;
        }
        
        for(int i=0; i<iArr1.length; i++) {
            System.out.print(iArr1[i]+",");
        }
        
        System.out.println();
        System.out.println(Arrays.toString(iArr2));
        System.out.println(Arrays.toString(iArr3));
        System.out.println(Arrays.toString(chArr));
        System.out.println(iArr3);
        System.out.println(chArr);
    }
}
```

- 실행 결과

  1,2,3,4,5,6,7,8,9,10

  [3, 4, 8, 10, 1, 10, 6, 2, 7, 1]	//랜덤한 숫자

  [100, 95, 80, 70, 60]

  [a, b, c, d]

  [I@14318bb		//문자를 제외하고는 index값이 추출된다.

  abcd		//문자값은 그대로 추출된다(문자열은 제외)

---

- 예제 5-3 (배열의 복사)

```java
class ArrayEx3 {
    public static void main(String[] args) {
        int[] arr = new int[5];
        
        for(int i=0; i<arr.length; i++){
            arr[i] = i+1;
        }
        
        System.out.println("[변경전]");
        System.out.println("arr.length:"+arr.length);
        for(int i=0; i<arr.length; i++)
            System.out.println("arr["+i+"]:"+arr[i]);
        
        int[] tmp = new int[arr.length*2];
        
        for(int i=0; i<arr.length; i++)
            tmp[i] = arr[i];
        
        arr = tmp;
        
        System.out.println("[변경후]");
        System.out.println("arr.length:"+arr.length);
        for(int i=0; i<arr.length; i++)
            System.out.println("arr["+i+"]:"+arr[i]);
    }
}
```

- 실행 결과

  [변경전]

  arr.length:5

  arr[0]:1

  arr[1]:2

  arr[2]:3

  arr[3]:4

  arr[4]:5

  [변경후]

  arr.length:10

  arr[0]:1

  arr[1]:2

  arr[2]:3

  arr[3]:4

  arr[4]:5

  arr[5]:0

  arr[6]:0

  arr[7]:0

  arr[8]:0

  arr[9]:0

---

- 예제 5-4(System.arraycopy()를 이용한 배열 복사)

```java
class ArrayEx4 {
    public static void main(String[] args){
        char[] abc = {'A','B','C','D'};
        char[] num = {'0','1','2','3','4','5','6','7','8','9'};
        System.out.println(abc);
        System.out.println(num);
        //배열 abc와 num을 붙여서 하나의 배열 result로 만든다.
        char[] result = new char[abc.length+num.length];
        System.arraycopy(abc, 0, result, 0, abc.length);
        System.arraycopy(num, 0, result, abc.length, num.length);
        System.out.println(result);
        //배열 abc를 배열 num의 첫 번째 위치부터 배열 abc의 길이만큼 복사
        System.arraycopy(abc, 0, num, 0, abc.length);
        System.out.println(num);
        //number의 인덱스 6 위치에 3개를 복사
        System.arraycopy(abc, 0, num, 6,3);
        System.out.println(num);
    }
}
```

- 실행 결과

  ABCD

  0123456789

  ABCD0123456789

  ABCD456789

  ABCD45ABC9

- System.arraycopy(복사하려는 배열, 시작 위치, 붙여넣으려는 배열, 붙여넣을 시작위치, 복사할 개수)

---

- 예제 5-5

```java
class ArrayEx5 {
    public static void main(String[] args) {
        int sum =0;
        float average = 0;
        
        int[] score = {100, 88, 100, 100, 90};
        
        for(int i=0; i<score.length; i++) {
            sum += score[i];
        }
        average = sum/(float)score.length;	//계산결과를 float로 얻기 위해 형변환
        
        System.out.println("총점 : "+sum);
        System.out.println("평균 : "+average);
    }
}
```

- 실행 결과

  총점 : 478

  평균 : 95.6

---

- 예제 5-6

```java
class ArrayEx6 {
    public static void main(String[] args) {
        int[] score = {79, 88, 91, 33, 100, 55, 95};
        
        int max = score[0];
        int min = score[0];
        
        for(int i=1; i<score.length; i++) {
            if(score[i] > max) {
                max = score[i];
            } else if(score[i] < min) {
                min = score[i];
            }
        }
        
        System.out.println("최대값 : "+max);
        System.out.println("최소값 : "+min);
    }
}
```

- 실행 결과

  최대값 : 100

  최소값 : 33

---

- 예제 5-7

```java
class ArrayEx7 {
    public static void main(String[] args) {
        int[] numArr = new int[10];
        
        for(int i=0; i<numArr.length; i++) {
            numArr[i] = i;
            System.out.print(numArr[i]);
        }
        System.out.println();
        
        for(int i=0; i<100; i++) {
            int n = (int)(Math.random()*10);
            int tmp = numArr[0];
            numArr[0] = numArr[n];
            numArr[n] = tmp;
        }
        for(int i=0; i<numArr.length ; i++)
            System.out.print(numArr[i]);
    }
}
```

- 실행 결과

  0123456789

  7319058264

- numArr을 뒤섞는 효과

---

- 예제 5-8

```java
class ArrayEx8 {
    public static void main(String[] args) {
        int[] ball = new int[45];
        
        for(int i=0; i<ball.length; i++)
            ball[i] = i+1;
        
        int temp = 0;
        int j = 0;
        
        for(int i=0; i<6; i++) {
            j=(int)(Math.random() * 45);
            temp = ball[i];
            ball[i] = ball[j];
            ball[j] = temp;
        }
        for(int i=0; i<6; i++)
            System.out.printf("ball[%d]=%d%n", i, ball[i]);
    }
}
```

- 실행 결과

  ball[0]=12

  ball[1]=40

  ball[2]=19

  ball[3]=39

  ball[4]=29

  ball[5]=3

---

- 예제 5-9

```java
import java.util.*;
class ArrayEx9 {
    public static void main(String[] args) {
        int[] code = {-4, -1, 3, 6, 11};
        int[] arr = new int[10];
        
        for(int i=0; i<arr.length; i++) {
            int tmp = (int)(Math.random() * code.length);
            arr[i] = code[tmp];
        }
        System.out.println(Arrays.toString(arr));
    }
}
```

- 실행 결과

  [-4, -4, -1, -1, 3, 6, 3, 3, 11, 3]

---

- 예제 5-10

```java
class ArrayEx10 {
    public static void main(String[] args) {
        int[] numArr = new int[10];
        
        for(int i=0; i<numArr.length; i++) {
            System.out.print(numArr[i] = (int)(Math.random() * 10));
        }
        System.out.println();
        
        for(int i=0; i<numArr.length-1 ; i++) {
            boolean changed = false;
            
            for(int j=0; j<numArr.length-1-i; j++) {
                if(numArr[j]>numArr[j+1]) {
                    int temp = numArr[j];
                    numArr[j] = numArr[j+1];
                    numArr[j+1] = temp;
                    changed = true;
                }
            }
            if(!changed) break;
            
            for(int k=0; k<numArr.length; k++)
                System.out.print(numArr[k]);
            System.out.println();
        }
    }
}
```

- 실행 결과

  1344213843

  1342134438

  1321344348

  1213343448

  1123334448

- 오름차순 배열이 되어있음.

---

- 예제 5-11

```java
class ArrayEx11 {
    public static void main(String[] args) {
        int[] numArr = new int[10];
        int[] counter = new int[10];
        
        for(int i=0; i<numArr.length; i++) {
            numArr[i] = (int)(Math.random()*10);
            System.out.print(numArr[i]);
        }
        System.out.println();
        
        for(int i=0; i<numArr.length; i++) {
            counter[numArr[i]]++;
        }
        
        for(int i=0; i<numArr.length; i++) {
            System.out.println(i+"의 개수 :"+counter[i]);
        }
    }
}
```

- 실행 결과

  4446579753

  0의 개수 : 0

  0의 개수 : 0

  0의 개수 : 0

  0의 개수 : 1

  0의 개수 : 3

  0의 개수 : 2

  0의 개수 : 1

  0의 개수 : 2

  0의 개수 : 0

  0의 개수 : 1

