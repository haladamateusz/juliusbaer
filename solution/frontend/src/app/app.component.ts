import {
  AfterViewInit,
  ChangeDetectorRef,
  Component,
  ElementRef,
  inject,
  ViewChild
} from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AsyncPipe, DatePipe, NgClass, NgOptimizedImage } from '@angular/common';
import { delay, interval, Observable, of, switchMap, takeWhile, timer } from 'rxjs';
import { Oscilloscope } from '@teropa/oscilloscope';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NgOptimizedImage, AsyncPipe, DatePipe, NgClass],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements AfterViewInit {
  title = 'd-fakeit';

  callDuration$: Observable<number> = timer(0, 1000);

  inputString =
    "Good morning. I'm Amina Curri born on 23rd November 1991. Social Security Number seven five six one two three four" +
    ' five six seven eight nine zero. (7561234567890) I need to withdraw a significant amount from my account in a small untraceable' +
    ' demilination. Can you manage that today?';

  importantData = {
    name: 'Amina Kouri',
    dateOfBirth: '23rd November 1991',
    ssn: '7561234567890'
  };

  dataName = {
    name: 'Amina Kouri',
    profession: 'Tech Startup Founder',
    validityStatus: 'unknown'
  };

  dataFields = [
    {
      name: 'Birthday',
      value: '23rd November 1991',
      validityStatus: 'unknown'
    },
    {
      name: 'Marital status',
      value: 'Single',
      validityStatus: 'unknown'
    },
    {
      name: 'Account Number',
      value: 'ZR1023',
      validityStatus: 'unknown'
    },
    {
      name: 'Tax Residency',
      value: ' USA',
      validityStatus: 'unknown'
    },
    {
      name: 'Net worth in milions',
      value: "$250M",
      validityStatus: 'unknown'
    },
    {
      name: 'Social Security Number',
      value: '7561234567890',
      validityStatus: 'unknown'
    },
    {
      name: 'Relationship manager',
      value: 'Ella Morrison',
      validityStatus: 'unknown'
    },
    {
      name: 'Education',
      value: 'Bachelor of Science in Computer Science',
      validityStatus: 'unknown'
    }
  ];

  clientNameAuthenticated = false;

  importantDataArray = ['Amina', 'Curri', '23rd', 'November', '1991.', '(7561234567890)'];

  emissions = 0;

  isEmitting = false;

  impersonatorCheckStatus = 'unknown';

  deepFakeCheckStatus = 'unknown';

  clientIndetificationCheckStatus = 'unknown';

  confidence = 0;

  @ViewChild('canvasWrapper') canvasWrapper!: ElementRef;

  processingFinished = false;

  textInput$ = interval(550).pipe(
    delay(2000),
    takeWhile(() => this.emissions < this.inputString.split(' ').length, true),
    switchMap((value: number) => {
      const correctInformations = this.dataFields.map(
        (field) => field.validityStatus === 'valid'
      ).length;

      if (value % 3 === 0) {
        this.confidence += Math.floor(Math.random() * 3);
      }

      if (value === 4) {
        this.deepFakeCheckStatus = 'valid';
        this.confidence += 20;
      }
      if (value === 12) {
        this.impersonatorCheckStatus = 'valid';
        this.confidence += 20;
      }
      if (value === 18) {
        this.clientIndetificationCheckStatus = 'invalid';
      }
      if (this.emissions < this.inputString.split(' ').length) {
        this.isEmitting = true;
        this.emissions++;
        return of(this.inputString.split(' ').slice(0, this.emissions));
      } else {
        this.isEmitting = false;
        this.processingFinished = true;
        return of(this.inputString.split(' '));
      }
    })
  );

  async ngAfterViewInit() {
    const audio = document.getElementById('audio') as HTMLAudioElement;

    const audio2 = document.getElementById('audio2') as HTMLAudioElement;
    audio2.src = '4USM1D1I1Q.wav';
    audio2.load();
    audio2.play();

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;

    audio.src = '4USM1D1I1Q.wav';
    audio.load();
    audio.play();

    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaElementSource(audio);

    let oscilloscope: Oscilloscope = new Oscilloscope(canvas, audioCtx, {
      fftSize: 16384,
      backgroundColor: '#f4f4f4',
      lineWidth: 2
    });
    oscilloscope.connect(source);
    oscilloscope.start();
    canvas.width = this.canvasWrapper.nativeElement.offsetWidth;
  }

  checkImportantData(word: string): boolean {
    if (word === 'Security') {
      this.clientNameAuthenticated = true;
    }
    if (this.importantDataArray.includes(word)) {
      if (word === '(7561234567890)') {
        this.dataFields[5].validityStatus = 'valid';
      }

      if (word === '1991.') {
        this.dataFields[0].validityStatus = 'valid';
      }
      if (word === 'Curri') {
        this.dataName.validityStatus = 'invalid';
      }
      return true;
    }
    return false;
  }
}
