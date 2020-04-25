# 자바의 정석 예제 곱씹어보기

## 4. 조건문과 반복문

### 반복문 - for, while, do-while

- 예제 4-12

```java
class FlowEX12 {
    public static void main(String[] args) {
        for(int i=1 ; i<=5 ; i++)
            System.out.pringln(i);
        
        for(int i=1 ; i<5 ; i++)
            System.out.pring(i);
        
        System.out.println();
    }
}
```

- 실행 결과

  1

  2

  3

  4

  5

  12345

---

- 예제 4-13

```java
class FlowEx13 {
    public static void main(String[] args) {
        int sum = 0;
        
        for(int i=1 ; i<=10 ; i++) {
            sum +=1 ;
            System.out.printf("1부터 %2d 까지의 합 : %2%n", i, sum);
        }
    }
}
```

- 실행 결과

  1부터 1 까지의 합 : 1

  1부터 2 까지의 합 : 3

  1부터 3 까지의 합 : 6

  1부터 4 까지의 합 : 10

  1부터 5 까지의 합 : 15

  1부터 6 까지의 합 : 21

  1부터 7 까지의 합 : 28

  1부터 8 까지의 합 : 36

  1부터 9 까지의 합 : 45

  1부터 10 까지의 합 : 55

---

- 예제 4-14

```java
class FlowEx14 {
    public static void main(String[] args) {
        for(int i=1 , j=10; i<=10 ; i++,j--)
            System.out.printf("%d \t %d%n", i, j);
    }
}
```

- 실행 결과

  1	10

  2	9

  3	8

  4	7

  5	6

  6	5

  7	4

  8	3

  9	2

  10	1

---

- 예제 4-15

```java
class FlowEx15 {
	public static void main(String[] args) {
        System.out.println("i \t 2*i \t 2*i-1 \t i*i \t 11-i \t i%3 \t i/3");
        System.out pirntln("-----------------------");
        
        for(int i=1 ; i<=10  ; i++)
            System.out.printf("%d \t %d \t %d \t %d \t %d \t %d \t %d%n", i, 2*i, 2*i-1, i*i, 11-i, i%3, i/3);
    }
}
```

- 실행 결과

  i		2*i		2*i-1		i*i		11-i		i%3		i/3

  \--------------------------------------------------------------------

  1		2		1			1		 10			 1			0

  2		4		3			4		 9				2			0

  3		6		5			9		 8				0			1

  4		8		7			16	   7				1			1

  5		10	  9			25	   6				2			1

  6		12	  11	  	36	   5				0			2

  7		14	  13	  	49	   4				1			2

  8		16	  15	  	64	   3				2			2

  9		18	  17	  	81	   2				0			3

  10	  20	  19	  	100	 1				1			3

---

- 예제 4-16

```java
class FlowEx16 {
    public static void main(String[] args) {
        for(int i=1 ; i<=5 ; i++) {
            for(int j=1 ; j<=10 ; j++) {
                System.put.print("^");
            }
            System.out.println();
        }
    }
}
```

- 실행 결과

  ^^^^^^^^^^

  ^^^^^^^^^^

  ^^^^^^^^^^

  ^^^^^^^^^^

  ^^^^^^^^^^

---

- 예제 4-17

```java
import java.util.*;

class FlowEx17 {
    public static void main(String[] args) {
        int num = 0;
        
        System.out.print("^을 출력할 라인의 수를 입력하세요.>");
        
        Scanner scanner = new Scanner(System.in);
        String tmp = scanner.nextLine();
        num = Integer.parseInt(tmp);
        
        for(int i=0 ; i<num ; i++) {
            for(int j=0 ; j<=1 ; j++) {
                System.out.print("^");
            }
            System.out.println();
        }
    }
}
```

- 실행 결과

  ^

  ^^

  ^^^

  ^^^^

  ^^^^^

  ^^^^^^

  ^^^^^^^

  ^^^^^^^^

  ^^^^^^^^^

  ^^^^^^^^^^

---

- 예제 4-18

```java
class FlowEx18 {
    public static void main(String[] args) {
        for(int i=2 ; i<=9 ; i++) {
            for(int j=1; j<=9 ; j++) {
                System.out.printf("%d x %d = %d%n", i,j,i*j);
            }
        }
    }
}
```

- 실행 결과

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  2 x 1 = 2

  3 x 1 = 3

  3 x 2 = 6

  ...

  9 x 9 = 81

---

- 예제 4-19

```java
class FlowEx19 {
    public static void main(String[] args) {
        for(int i=1; i<=3 ; i++) {
            for(int j=1 ; j<=3 ; j++) {
                for(int k=1 ; k<=3 ; k++) {
                    System.out.println(""+i+j+k);
                }
            }
        }
    }
}
```

- 실행 결과

  111

  112

  113

  121

  122

  ...

  332

  333

---

- 예제 4-21

```java
class FlowEx20 {
    public static void main(String[] args) {
        for(int i=1 ; i<=5 ; i++) {
            for(int j=1 ; j<=5 ; j++) {
                if(i==j) {
                    System.out.printf("[%d, %d]", i, j);
                } else {
                    System.out.printf("%5c", ' ');
                }
            }
            System.out.println();
        }
    }
}
```

- 실행 결과

  [1,1]

  ​	[2,2]

  ​		[3,3]

  ​			[4,4]

  ​				[5,5]

---

- 예제 4-22

```java
class FlowEx22 {
    public static void main(String[] args) {
        int[] arr = {10, 20, 30, 40, 50};
        int sum = 0;
        
        for(int i=0; i<arr.length; i++) {
            System.out.printf("%d ", arr[i]);
        }
        System.out.println();
        
        for(int tmp : arr) {
            System.out.printf("%d ", tmp);
            sum += tmp;
        }
        System.out.println();
        System.out.println("sum="+sum);
    }
}
```

- 실행 결과

  10 20 30 40 50

  10 20 30 40 50

  sum=150

---

- 예제 4-23

```java
class FlowEx23 {
    public static void main(String[] args) {
        int i=5;
        
        while(i--!=0) {
            System.out.println(i + " - I can do it.");
        }
    }
}
```

- 실행 결과

  4 - I can do it.

  3 - I can do it.

  2 - I can do it.

  1 - I can do it.

  0 - I can do it.

---

- 예제 4-24

```java
class FlowEx24 {
	public static void main(String[] args) {
        int i = 11;
        
        System.out.println("카운트 다운을 시작합니다.");
        
        while(i-- != 0) {
            System.out.println(i);
            
            for(int j=0 ; j<2_000_000_000;j++) {
                ;	//아무런 내용도 없는 빈 문장. 그저 조건식과 증감식을 2.000.000.000번 반복하며 시간을 보냄.		2_000_000_000이 무슨 내용인지 잘 모르겠음.
            }
        }
        System.out.println("GAME OVER")
    }
}
```

- 실행 결과

  카운트 다운을 시작합니다.

  10

  9

  8

  7

  6

  5

  4

  3

  2

  1

  GAME OVER

---

- 예제 4-25

```java
import java.util.*;

class FlowEx25 {
    public static void main(String[] args) {
        int num = 0, sum = 0;
        System.out.print("숫자를 입력하세요.");
        
        Scanner scanner = new Scanner(System.in);
        String tmp = scanner.nextLine();
        num = Integer.parseInt(tmp);
        
        while(num !=0) {
            sum += num%10;	// sum = sum + num%10;
            System.out.printf("sum=%3d num num=%d%n", sum, num); //%3d = 자릿수를 지정하고 오른쪽 정렬
            
            num /= 10;	// num = num / 10;
        }
        System.out.println("각 자리수 합 :"+sum);
    }
}
```

- 실행 결과

  숫자를 입력하세요.

  sum =   5	num = 12345

  sum =   9	num = 1234

  sum =  12	num = 123

  sum =  14	num = 12

  sum =   15	num = 1

  각 자리수의 합 : 15

---

- 예제 4-26

```java
class FlowEx26
    public static void main(String[] args) {
    int sum = 0;
    int i	= 0;
    
    while((sum += ++i) <= 20) {
        System.out.printf("%d - %d%n", i, sum);
    }
}
```

- 실행 결과

  1 - 1

  2 - 3

  3 - 6

  4 - 10

  5 - 15

---

- 예제 4-27

```java
import java.util.*;

class FlowEx27 {
    public static void  main(String[] args) {
        int num;
        int sum = 0;
        boolean flag = true;
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("합계를 구할 숫자를 입력하세요.(끝내려면 0을 입력)");
        
        while(flag) {
            System.out.print(">>");
            
            String tmp = Scanner.nextLine();
            num = Integer.parseInt(tmp);
            
            if(num !=0) {
                sum += num;
            } else {
                flag = false;
            }
        }
        System.out.println("합계 : "+sum);
    }
}
```

- 실행 결과

  합계를 구할 숫자를 입력하세요.(끝내려면 0을 입력)

  \>\>100

  \>\>200

  \>\>300

  \>\>400

  \>\>0

  합계 : 1000

---

- 예제 4-28

```java
import java.util.*;

class FlowEx28 {
    public static void main(String[] args) {
        int input = 0, answer = 0;
        
        answer = (int)(Math.random() * 100) + 1;
        Scanner scanner = new Scanner(System.in);
        
        do {
            System.out.print("1과 100사이의 정수를 입력하세요.>");
            input = scanner.nextInt();
            
            if(input > answer) {
                System.out.println("더 작은 수로 다시 시도해보세요.");
            } else if (input < answer) {
                System.out.println("더 큰 수로 다시 시도해보세요.");
            }
        } while(input != answer);
        System.out.println("정답입니다."){};
    }
}	//do-while문은 do 안의 내용을 최소 한번은 수행한 뒤 while문을 돌리게 된다.
```

- 실행 결과

  1과 100사이의 정수를 입력하세요.>50

  더 작은 값으로 다시 시도해보세요.

  1과 100사이의 정수를 입력하세요.>12

  더 큰 값으로 다시 시도해보세요.

  1과 100사이의 정수를 입력하세요.>21

  정답입니다.

---

- 예제 4-29

```java
class FlowEx29 {
    public static void main(String[] args) {
        for(int i=1 ; i<=100 ; i++) {
            System.out.printf("i=%d ", i);
        
            int tmp = i;
            
            do {
                if(tmp%10%3 == 0 && tmp%10 !=0){
                    System.out.print("짝");
                }
            } while((tmp/=10) !=0){};
            System.out.println();
        }
    }
}
```

- 실행 결과

  i=1

  i=2

  i=3 짝

  i=4

  i=5

  i=6 짝

  ...

  i=98 짝

  i=99 짝짝

  i=100

---

- 예제 4-30

```java
class FlowEx30 {
    public static void main(String[] args) {
    	int sum = 0;
    	int i	= 0;
    
    	while(true) {
	        if(sum > 100)
	            break;
        
	        ++i;
	        sum += i;
	    }
        
        System.out.println("i="+i);
        System.out.println("sum="+sum);
	}
}
```

- 실행 결과

  i=14

  sum=105

---

- 예제 4-31

```java
class FlowEx31 {
    public static void main(String[] args) {
        for(int i=0; i<=10; i++) {
            if(i%3 == 0)
                continue;
            
            System.out.println(i);
        }
    }
}
```

- 실행 결과

  1

  2

  4

  5

  7

  8

  10

---

- 예제 4-32

```java
import java.util.*;

class FlowEx32 {
    public static void main(String[] args) {
        int menu = 0;
        int num = 0;
        
        Scanner scanner = new Scanner(System.in);
        
        while(true) {
            System.out.println("(1) square");
            System.out.println("(2) square root");
            System.out.println("(3) log");
            System.out.println("원하는 메뉴 (1~3)를 선택하세요.(종료:0)>");
            
            String tmp = scanner.nextLine();
            menu = Integer.parseInt(tmp);
            
            if(menu==0) {
                System.out.println("프로그램을 종료합니다.");
                break;
            } else if (!(1<= menu && menu <=3)) {
                System.out.println("메뉴를 잘못 선택하셨습니다.(종료는 0)");
                continue;
            }
            System.out.println("선택하신 메뉴는 "+menu+"번입니다.");
        }
    }
}
```

---

- 예제 4-33

```java
class FlowEx33 {
    public static void main(String[] args) {
        Loop1 : for(int i=2 ; i<=9 ; i++) {
            for(int j=1 ; j<=9 ; j++) {
                if(j==5) {
                    break Loop1;
                }
                System.out.println(i+"*"+j+"="+(i*j));
            }
        }
    }
}	//반복문에 이름을 붙여 주고 break문에 반복문 이름을 지정해주면 하나 이상의 반복문도 벗어날 수 있다.
```

- 실행 결과

  2*1 = 2

  2*2 = 4

  2*3 = 6

  2*4 = 8

