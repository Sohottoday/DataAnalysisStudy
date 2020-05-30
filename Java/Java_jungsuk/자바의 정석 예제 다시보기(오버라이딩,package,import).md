# 자바의 정석 예제 다시보기

## 객체지향프로그래밍

### 오버라이딩(overriding)

- 예제 7-5

```java
class SuperTest {
    public static void main(String args[]) {
        Child c = new Child();
        c.method();
    }
}
```

```java
class Parent {
    int x = 10;
}
```

```java
class Child extends Parent {
    void method() {
        System.out.println("x=" +x);
        System.out.println("this.x=" + this.x);
        System.out.println("super.x=" + super.x);
    }
}
```

- 실행결과

  x=10

  this.x=10

  super.x=10

---

- 예제 7-6

```java
class Parent {
    int x = 10;
}
```

```java
class Child extends Parent {
    int x = 20;
    
    void method() {
        System.out.println("x=" + x);
        System.out.println("this.x=" + this.x);
        System.out.println("super.x=" + super.x);
    }
}
```

```java
class SuperTest2 {
    public static void main(String args[]) {
        Child c = new Child();
        c.method();
    }
}
```

- 실행결과

  x=20

  this.x=20

  super.x=10

---

- Object클래스를 제외한 모든 클래스의 생성자 첫 줄에 생성자, this(), super()를 호출해야 한다. 그렇지 않으면 컴파일러가 자동적으로 'super();'를 생성자의 첫 줄에 삽입한다.

---

- 예제 7-7

```java
class Point {
    int x, y;
    
    Point(int x, int y) {
        this.x = x;
        this.y = y;
    }
    
    String getLocation() {
        return "x : " + x + ",y : " + y;
    }
}
```

```java
class Point3D extends Point {
    int z;
    
    Point3D(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    String getLocation() {
        return "x : " + x + ",y : " + y + ", z : " + z;
    }
}
```

```java
class PointTest {
    public static void main(String args[]) {
        Point3D p3 = new Point3D(1,2,3);
    }
}
```

- 실행결과

  컴파일 에러 발생

  Point3D 클래스의 생성자에서 조상 클래스의 생성자인 Point()를 찾을 수 없다는 내용

  Point3D클래스의 생성자의 첫 줄이 생성자(조상의 것이든 자신의 것이든)를 호출하는 문장이 아니기 때문에 컴파일러는 다음과 같이 자동적으로 'super();'를 Point3D 클래스의 생성자의 첫 줄에 넣어준다.

---

- 예제 7-8

```JAVA
class Point {
    int x = 10;
    int y = 20;
    
    Point(int x, int y) {
        this.x = x;
        this.y = y;
    }
}
```

```java
class Point3D extends Point {
    int z = 30;
    
    Point3D() {
        this(100, 200, 300);		//Point3D(int x, int y, int z)를 호출한다
    }
    
    Point3D(int x, int y, int z) {
        super(x, y);		//Point(int x, int y)를 호출한다.
        this.z = z;
    }
}
```

```java
class PointTest2 {
    public static void main(String args[]) {
        Point3D p3 = new Point3D();
        System.out.println("p3.x=" + p3.x);
        System.out.println("p3.y=" + p3.y);
        System.out.println("p3.z=" + p3.z);
    }
}
```

- 실행결과

  p3.x=100

  p3.y=200

  p3.z=300

---

- 예제 7-9

```java
package com.codechobo.book;

class PackageTest {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
```

---

- 예제 7-10

```java
import java.text.SimpleDateFormat;
import java.util.Date;

class ImportTest {
    public static void main(String[] args) {
        Date today = new Date();
        
        SimpleDateFormat date = new SimpleDateFormat("yyyy/MM/dd");
        SimpleDateFormat time = new SimpleDateFormat("hh:mm:ss a");
        
        System.out.println("오늘 날짜는 " + date.format(today));
        System.out.println("현재 시간은 " + time.format(today));
    }
}
```

- 실행결과

  오늘 날짜는 2020/05/30

  현재 시간은 07:52:53 오후

---

- static import문을 선언하면 간략하게 표현 가능
- 예제 7-11

```java
import static java.lang.System.out;
import static java.lang.Math.*;

class StaticImportEx1 {
    public static void main(String[] args) {
        out.println(random());		//System.out.println(Math.random());
        
        out.println("Math.PI : " + PI);		//System.out.println("Math.PI : " + Math.PI);
    }
}
```

- 실행결과

  0.63727768215...

  Math.PI : 3.1415926535...