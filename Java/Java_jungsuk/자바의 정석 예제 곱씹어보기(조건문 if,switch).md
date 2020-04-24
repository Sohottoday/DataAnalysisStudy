# 자바의 정석 예제 곱씹어보기

## 4. 조건문과 반복문

### 조건문  -if, switch

- if(조건식) {

  //조건이 참(true)일 때 수행될 문장들을 적는다.

  }

- 예제 4-1

```java
class FlowEx1 {
    public static void main(string[] args) {
        int x=0;
        System.out.printf("x=%d 일 때, 참인 것은 %n", x);
        
        if(x == 0) System.out.println("x==0");
        if(x != 0) System.out.println("x!=0");
        if(!(x == 0)) System.out.println("!(x==0)");
        if(!(x != 0)) System.out.println("!(x!=0)");
        
        x = 1;
        System.out.printf("x=%d 일 때, 참인 것은 %n", x);
        
        if(x == 0) System.out.println("x==0");
        if(x != 0) System.out.println("x!=0");
        if(!(x == 0)) System.out.println("!(x==0)");
        if(!(x != 0)) System.out.println("!(x!=0)");
    }
}

```

- 실행 결과

  x=0 일 때, 참인 것은

  x==0

  !(x!=0)

  x=1 일 때, 참인 것은

  x!=0

  !(x==0)

---

- 예제 4-2

```java
import java.util.*; // Scanner클래스를 사용하기 위해 추가

class FlowEx2 {
    public static void main(String[] args){
        int input;
        
        System.out.print("숫자를 하나 입력하세요.>");
        
        Scanner scanner = new Scanner(System.in);]
        String tmp - scanner.nextLine();	//화면을 통해 입력받은 내용을 tmp에 저장
        input = Integer.parseInt(tmp);	// 입력받은 문자열(tmp)을 숫자로 변환
        
        if(input==0) {
            System.out.println("입력하신 숫자는 0입니다.");
        }
        
        if(input !=0) {
            System.out.println("입력하신 숫자는 0이 아닙니다.");
            System.out.println("입력하신 숫자는 %d입니다.", input);
        }
    }
}
```

- 실행 결과 1

  숫자를 하나 입력하세요. > 3

  입력하신 숫자는 0이 아닙니다.

  입력하신 숫자는 3 입니다.

- 실행 결과 2

  숫자를 하나 입력하세요. > 0

  입력하신 숫자는 0입니다.

  입력하신 숫자는 0입니다.

---

- 예제 4-3

```java
import java.util.*; //Scanner클래스를 사용하기 위해 추가

class FlowEx3 {
    public static void main(String[] args) {
        System.out.println("숫자를 하나 입력하세요.>");
        
        Scanner scanner = new Scanner(System.in);
        int input = scanner.nextInt();
        
        if(input==0) {
            System.out.println("입력하신 숫자는 0입니다");
        } else {	//input != 0 인 경우
            System.out.println("입력하신 숫자는 0이 아닙니다.");
        }
    }
}
```

- 실행 결과 1

  숫자를 하나 입력하세요. > 3

  입력하신 숫자는 0이 아닙니다.

- 실행 결과 2

  숫자를 하나 입력하세요. > 0

  입력하신 숫자는 0입니다.

---

- 예제 4-4

```java
import java.util.*;

class FlowEx4 {
    public static void main(String[] args) {
        int score = 0;	//점수를 저장하기 위한 변수
    	char grade = ' ';	//학점을 저장하기 위한 변수. 공백으로 초기화한다.
    
    	System.out.print("점수를 입력하세요.>");
    	Scanner scanner = new Scanner(System.in);
    	score = scanner.nextInt();	//화면을 통해 입력받은 숫자를 score에 저장
    
    	if ( score >= 90) {
	        grade = 'A';
	    } else if (score >=80) {
	        grade = 'B';
	    } else if (score >=70) {
	        grade = 'C';
	    } else {
	        grade = 'D';
	    }
	    System.out.println("당신의 학점은 "+ grade + "입니다.");
	}
}
```

- 실행 결과 1

  점수를 입력하세요.> 70

  당신의 학점은 C입니다.

- 실행 결과 2

  점수를 입력하세요.>63

  당신의 학점은 D입니다.

---

- 예제 4-5

```JAVA
import java.util.*;

class FlowEx5 {
    public static void main(String[] args) {
        int score = 0;
        char grade = ' ', opt = '0';
        
        System.out.print("점수를 입력해주세요.>");
        
        Scanner scanner = new scanner(System.in);
        score = scanner.nextInt();
        
        System.out.printf("당신의 점수는 %d입니다. %n", score);
        
        if(score >=90) {
            grade = 'A';
            if(scroe >=98) {
                opt = '+';
            } else if (score <94 {
                ort = '-';
            })
        } else if (score >=80) {
            grade = 'B';
            if (score >=88) {
                opt = '+'
            } else if (score < 84) {
                opt = '-';
            }
        } else {
            grade = 'C';
        }
        System.out.printf("당신의 학점은 %c%c입니다. %n", grade, opt);
    }
}

```

- 실행 결과 1

  점수를 입력해주세요.>100

  당신의 점수는 100입니다.

  당신의 학점은 A+입니다.

- 실행 결과2

  점수를 입력해 주세요.>85

  당신의 점수는 81입니다.

  당신의 학점은 B0입니다.

---

- 예제 4-6

```java
import java.util.*;

class FlowEx6 {
    public static void main(String[] args) {
        System.out.print("현재 월을 입력하세요.>");
        
        Scanner scanner = new Scanner(System.in);
        int month = scanner.nextInt();
        
        switch(month) {
            case 3:
            case 4:
            case 5:
                System.out.println("현재의 계절은 봄입니다.");
                break;
            case 6: case 7: case 8:
                System.out.println("현재의 계절은 여름입니다");
                break;
            case 8: case 10: case 11:
                System.out.println("현재의 계절은 가을입니다.");
                break;
            default:
              	System.out.println("현재의 계절은 겨울입니다.");
        }
    }
}
```

- 실행 결과

  현재 월을 입력하세요.>3

  현재의 계절은 봄입니다.

---

- 예제 4-7

```java
import java.util.*;

class FlowEX7 {
    public static void main(String[] args) {
        System.out.print("가위(1), 바위(2), 보(3) 중 하나를 입력하세요.");
        
        Scanner scanner = new Scanner(System.in);
        int user = scanner.nextInt();
        int com = (int)(Math.random() *3) +1;	//1,2,3중 랜덤 숫자 하나가 com에 저장됨
        
        System.out.println("당신은 "+user+"입니다.");
        System.out.println("컴은 "+com+"입니다.");
        
        switch(user-com) {
            case 2: case -1:
                System.out.println("당신이 졌습니다.");
                break;
            case 1: case -2:
                System.out.println("당신이 이겼습니다.");
                break;
            case 0:
                System.out.println("비겼습니다.");
                //마지막 문장이므로 break를 사용할 필요가 없다.
        }
    }
}
```

- 실행 결과 1

  가위(1), 바위(2), 보(3) 중 하나를 입력하세요.

  당신은 1입니다.

  컴은 1입니다.

  비겼습니다.

- 실행 결과 2

  가위(1), 바위(2), 보(3) 중 하나를 입력하세요.

  당신은 3입니다.

  컴은 2입니다.

  당신이 이겼습니다.

---

- 예제 4-8

```java
import java.util.*;
class FlowEx8 {
    public static void main(String[] args) {
        System.out.print("당신의 주민번호를 입력하세요.(011231-1111222)>");
        
        Scanner scanner = new Scanner(System.in);
        String regNo = scanner.nextLine();
        
        char gender = regNo.charAt(7);	//입력받은 번호의 8번째 문자를 gender에 저장
        
        switch(gender) {
            case '1': case '3':
                System.out.println("당신은 남자입니다.");
                break;
            case '2': case '4':
                System.out.println("당신은 여자입니다.");
                break;
            default:
                System.out.println("유효하지 않은 주민등록번호입니다.");
        }
    }
}
```

- 실행 결과

  당신의 주민번호를 입력하세요. (011231-1111222)> 110101-2111222

  당신은 여자입니다.

---

- 예제 4-9

```java
import java.util.*;

class FlowEx9 {
    public static void main(String[] args) {
        char grade = ' ';
        
        System.out.print("당신의 점수를 입력하세요.(1~100)");
        
        Scanner scanner = new Scanner(System.in);
        int score = scanner.nextInt();
        
        switch(score) {
            case 100: case 99: case 98: case 97: case 96:
            case 95: case 94: case 93: case 92: case 91: case 90:
                grade = 'A';
                break;
            case 89: case 88: case 87: case 86: case 85:
            case 84: case 83: case 82: case 81: case 80:
                grade = 'B';
                break;
            default :
                grade = 'F';
        }
        System.out.println("당신의 학점은 "+grade+"입니다.");
    }
}
```

- 실행 결과

  당신의 점수를 입력하세요.(1~100)> 82

  당신의 학점은 B입니다.

---

- 예제 4-10

```java
import java.util.*;

class FlowEx10 {
    public static void main(String[] args) {
        int score = 0;
        char grade = ' ';
        
        System.out.print("당신의 점수를 입력하세요.(1~100)>");
        
        Scanner scanner = new Scanner(System.in);
        String tmp = scanner.nextLine();
        score = Integer.parseInt(tmp);
        
        switch(score/10) {
            case 10 :
            case 9 :
                grade = 'A';
                break;
            case 8 :
                grade = 'B';
                break;
            case 7 :
                grade = "C";
                break;
            default :
                grade = 'F';
        }
        System.out.println("당신의 학점은 "+grade+"입니다.");
    }
}
```

- 실행 결과

당신의 점수를 입력하세요.(1~100)> 82

당신의 학점은 B입니다.

---

- 예제 4-11

```java
import java.util.*;

class FlowEx11 {
    public static void main(String[] args) {
        System.out.print("당신의 주민번호를 입력하세요.(011231-1111222)>");
        
        Scanner scanner = new Scanner(System.in);
        String regNo = scanner.nextLine();
        char gender = regNo.charAt(7);
        
        switch(gender) {
            case '1'; case'3':
                switch(gender) {
                    case '1':
                        System.out.println("당신은 2000년 이전에 출생한 남자입니다.");
                        break;
                    case '3':
                        System.out.println("당신은 2000년 이후에 출생한 남자입니다.");
                }
                break;	//이 break문을 빼먹지 않도록 주의
            case '2': case '4':
                switch(gender) {
                    case '2':
                        System.out.println("당신은 2000년 이전에 출생한 여자입니다.");
                        break;
                    case '4':
                        System.out.println("당신은 2000년 이후에 출생한 여자입니다.");
                        break;
                }
                break;
            default :
                System.out.println("유효하지 않은 주민등록번호입니다.");
        }
    }
}
```

- 실행 결과

  당신의 주민번호를 입력하세요.(011231-1111222)>010101-4111222

  당신은 2000년 이후에 출생한 여자입니다.