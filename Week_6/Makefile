test: clean
	python label.py
	mkdir test
	xcopy /E /I image test
	xcopy /E /I annotation test

clean:
	if exist annotation rmdir /s /q annotation
	if exist test rmdir /s /q test
