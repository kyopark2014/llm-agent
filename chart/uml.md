# State Diagram

[activity-diagram.plantuml](https://github.com/joelparkerhenderson/plantuml-examples/blob/master/doc/activity-diagram/activity-diagram.plantuml.txt)

```text
# State Diagram

@startuml
skinparam monochrome true
start
-> Starting;
:Activity 1;
if (Question) then (yes)
  :Option 1;
else (no)
  :Option 2;
endif
:Activity 2;
-> Stopping;
stop
@enduml
```

이때의 결과는 아래와 같습니다.

