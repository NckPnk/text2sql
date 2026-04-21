# Text2SQL (Windows 10: пошаговый запуск для начинающих)

Это инструкция «с нуля», чтобы запустить проект на **Windows 10** даже без большого опыта.

---

## Что делает приложение

Text2SQL принимает вопрос на русском языке и строит SQL-запрос к PostgreSQL.

Пример: «Покажи 10 лицевых счетов» → SQL + результат.

---

## 0) Что нужно установить заранее

Установите эти программы **по порядку**:

1. **Python 3.11+** (обязательно отметьте галочку **Add Python to PATH** при установке).
2. **Git for Windows**.
3. **Ollama for Windows**.
4. **PostgreSQL** (если у вас его ещё нет).

Проверка (откройте **PowerShell**):

```powershell
python --version
git --version
ollama --version
```

Если какая-то команда не найдена — перезапустите PowerShell/ПК после установки.

---

## 1) Откройте проект

Если проект уже скачан, перейдите в папку проекта:

```powershell
cd C:\path\to\text2sql
```

Если не скачан:

```powershell
git clone <URL_ВАШЕГО_РЕПО>
cd text2sql
```

---

## 2) Создайте виртуальное окружение и установите зависимости

В PowerShell из папки проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_current.txt
```

> Если появится ошибка про выполнение скриптов (`running scripts is disabled`), выполните:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> Закройте и заново откройте PowerShell, затем снова активируйте окружение.

---

## 3) Настройте `.env`

Скопируйте файл примера:

```powershell
copy .env.example .env
```

Откройте `.env` в Блокноте/VS Code и заполните минимум:

- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_SCHEMA`
- `OLLAMA_BASE_URL` (обычно `http://localhost:11434`)
- `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBED_MODEL`

Пример значений Ollama:

- `OLLAMA_LLM_MODEL=qwen3-coder:30b`
- `OLLAMA_EMBED_MODEL=qwen3-embedding:8b`

---

## 4) Запустите Ollama и загрузите модели

В отдельном окне PowerShell:

```powershell
ollama serve
```

В другом окне PowerShell:

```powershell
ollama pull qwen3-coder:30b
ollama pull qwen3-embedding:8b
```

> Если в `.env` указаны другие модели — загрузите именно их.

---

## 5) Подготовьте PostgreSQL

Убедитесь, что:

- PostgreSQL запущен;
- база `POSTGRES_DB` существует;
- пользователь имеет доступ;
- схема `POSTGRES_SCHEMA` существует и содержит ваши таблицы.

Если вы не уверены, попросите администратора БД дать вам:

- host,
- port,
- db name,
- username,
- password,
- schema.

---

## 6) Выполните индексацию схемы (обязательно перед первым запуском)

Из папки проекта с активным `.venv`:

```powershell
python scripts\index_schema.py
```

Индексация может занять несколько минут.

---

## 7) Запустите приложение

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Когда сервер запущен, откройте:

- UI: http://localhost:8000/static/index.html
- Health-check: http://localhost:8000/api/health

---

## 8) Как пользоваться

### Вариант A (самый простой): через UI

1. Откройте страницу `http://localhost:8000/static/index.html`
2. Введите вопрос на русском.
3. Нажмите кнопку отправки.

### Вариант B: через API (PowerShell)

```powershell
curl.exe -X POST http://localhost:8000/api/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Покажи 10 лицевых счетов\"}"
```

---

## 9) Быстрая диагностика

Базовая проверка:

```powershell
python scripts\smoke_test.py
```

Расширенная проверка:

```powershell
python scripts\smoke_test.py --full --api-url http://localhost:8000
```

---

## Частые проблемы и решения

1. **`python` не найден**
   - Переустановите Python и включите `Add Python to PATH`.

2. **Не запускается `.venv\Scripts\Activate.ps1`**
   - Выполните `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

3. **Ошибка подключения к Ollama**
   - Проверьте, что `ollama serve` запущена.
   - Проверьте `OLLAMA_BASE_URL` (обычно `http://localhost:11434`).

4. **Ошибка подключения к PostgreSQL**
   - Проверьте логин/пароль/порт/имя БД в `.env`.
   - Проверьте, что PostgreSQL-сервис работает.

5. **Слабый ПК / мало RAM для большой LLM**
   - Используйте более лёгкую модель в `.env` и загрузите её через `ollama pull`.

---

## Короткий чек-лист запуска

1. Активировать `.venv`
2. Проверить `.env`
3. Запустить `ollama serve`
4. Убедиться, что PostgreSQL доступен
5. Запустить `python scripts\index_schema.py`
6. Запустить `uvicorn app.main:app --host 0.0.0.0 --port 8000`
7. Открыть `http://localhost:8000/static/index.html`
