# 자바의 정석 예제 다시보기

## 배열

### String 배열

- 예제 5-12

```java
class ArrayEx12 {
    public static void main(String[] args) {
        String[] names = {"Kim", "Park", "Yi"};
        
        for(int i=0 ; i<names.length; i++)
            System.out.println("names["+i+"]"+names[i]);
        
        String tmp = names[2];
        System.out.println("tmp:"+tmp);
        names[0] = "Yu";
        
        for(String str : names)
            System.out.println(str);
    }
}
```

- 실행 결과

  names[0] : Kim

  names[1] : Park

  names[2] : Yi

  tmp:Yi

  Yu

  Park

  Yi

---

- 예제 5-13

```java
class ArrayEx13 {
    public static void main(String[] args) {
        char[] hex = {'C', 'A', 'F', 'E'};
        
        String[] binary = {"0000", "0001", "0010", "0011",
                          "0100", "0101", "0110", "0111",
                          "1000", "1001", "1010", "1011",
                          "1100", "1101", "1110", "1111"};
        
        String result = "";
        for(int i=0; i<hex.length; i++) {
            if(hex[i] >='0' && hex[i] <='9') {
                result += binary[hex[i]-'0'];
            } else {
                result += binary[hex[i]-'A'+10];
            }
        }
        System.out.println("hex:"+new String(hex));
        System.out.println("binary:"+result);
    }
}
```

- 실행 결과

  hex:CAFE

  binary:1100101011111110

---

- String 클래스 주요 메서드
  - char charAt(int index) : 문자열에서 해당 위치에 있는 문자를 반환
  - int length() : 문자열의 길이를 반환
  - String substring(int form, int to) : 문자열에서 해당 범위(from ~ to)에 있는 문자열을 반환(to 범위는 포함되지 않음)
  - boolean equals(Object obj) : 문자열의 내용이 obj와 같은지 확인한다.
  - char[] toCharArray() : 문자열을 분자배열(char[])로 변환해서 반환한다.

---

- 예제 5-14

```java
class ArrayEx14 {
    public static void main(String[] args) {
        String src = "ABCDE";
        
        for(int i=0; i<src.length(); i++) {
            char ch = src.charAt(i);
            System.out.println("src.charAt("+i+") : "+ch);
        }
        char[] chArr = src.toCharArray();
        
        System.out.println(chArr);
    }
}
```

- 실행 결과

  src.charAt(0) : A

  src.charAt(1) : B

  src.charAt(2) : C

  src.charAt(3) : D

  src.charAt(4) : E

  ABCDE

---

- 예제 5-15

```java
class ArrayEx15 {
    public static void main(String[] args) {
        String source = "SOSHELP";
        String[] morse = {
            ".-", "-...", "-.-.", "-..", ".",
            "..-.", "--.", "....", "..", ".---",
            "-.-", ".-..", "--", "-.", "---",
            ".--.", "--.-", ".-.", "...", "-",
            "..-", "...-", ".--", "-..-", "-.--", "--.."
        };
        String result="";
        
        for(int i=0; i<source.length(); i++) {
            result +=morse[source.charAt(i)-'A'];
        }
    }
}
```

- 실행 결과

  source : SOSHELP

  morse : ...---.........-...--.

---

