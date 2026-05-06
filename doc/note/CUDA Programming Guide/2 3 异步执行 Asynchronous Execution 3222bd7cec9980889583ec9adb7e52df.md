# 2.3 异步执行 Asynchronous Execution

# 异步并发执行 Asynchronous Concurrent Execution

CUDA 支持**多任务**的**并发（或称重叠）执行**，具体包括：

- 在主机上的计算
- 设备上的计算
- 从主机到设备的数据传输
- 从设备到主机的数据传输
- 给定设备内存内的数据传输
- 设备之间的内存传输

![图为使用 CUDA 流的异步并发执行](2%203%20%E5%BC%82%E6%AD%A5%E6%89%A7%E8%A1%8C%20Asynchronous%20Execution/62fbfd5b-2439-4a62-9aad-32a789553873.png)

图为使用 CUDA 流的异步并发执行

并发性通过**异步接口（API Calls）**实现，其中调度函数调用或内核启动会立即返回。

异步调用通常在已调度的操作**完成之前返回**，甚至可能在异步操作开始之前就返回。**【如图异步调用间隔比调度操作的间隔短】**

当需要获取**初始调度操作的最终结果**时，应用程序必须执行某种形式的**同步**以确保相关操作已完成。**【图中的synchronization point】**

并发执行的典型范例是主机与设备内存传输与计算任务的重叠执行，从而减少或消除其开销。

---

异步接口主要提供三种与已调度操作进行同步的方式：

- **阻塞式(blocking)方法**，其中应用程序调用一个会阻塞或等待操作完成的函数。
- **非阻塞式(non-blocking)方法**，或**轮询式方法(polling approach)**，其中应用程序调用一个立即返回并提供操作状态信息的函数。
- **回调式方法（callback）**，其中当操作完成时会**执行一个预先注册的函数**

CUDA 中异步执行的**核心 API 组件**是 **CUDA 流**与 **CUDA 事件**。

# CUDA流 CUDA Streams

流如同一个**工作队列**，程序可将**内存复制**或**内核启动**等操作按**顺序**加入队列等待执行。

工作流程：对于给定流而言，位于队列前端的操作会被执行并出列，使下一个排队操作移至前端等待执行。流中操作的执行顺序是连续的，且**严格按照它们加入流的顺序**执行。

**应用程序**可以**同时使用多个流**。在此情况下，运行时会根据 GPU 资源的状态，从具有可用任务的流中选择任务执行。

可以为**流分配优先级**，这作为运行时调度提示来影响执行顺序，但并**不保证特定的执行次序**。

CUDA 设有**默认流**，**未指定特定流**的操作与内核启动均会排队进入此默认流。未明确指定流的代码示例即隐式使用了该默认流。

## 创建与销毁CUDA流

CUDA 流可通过 **`cudaStreamCreate()`** 函数创建。该函数调用将**初始化流句柄**，该句柄可用于在后续函数调用中**标识对应流**。

```cpp
cudaStream_t stream; // Stream handle
cudaStreamCreate(&stream); // Create a new stream

// stream based operations

cudaStreamDestroy(stream);
```

<aside>
💡

如果应用程序在设备仍在流 `stream` 中执行工作时调用 **`cudaStreamDestroy()`** ，该流将在**销毁前完成流中的所有工作**。

</aside>

## 在CUDA流中启动内核

通常用于启动内核的**三重尖括号语法**也可以用于将内核启动到特定的流中。流被指定为内核启动的一个**额外参数**。内核启动是**异步**的，应用程序可以在内核执行期间在**CPU**或**GPU的其他流**中执行任务。

```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

<aside>
💡

名为 `kernel` 的内核被启动到句柄为 `stream` 的流中，该流属于 `cudaStream_t` 类型
且假定先前已被创建。

</aside>

## 在CUDA流中启动内存传输

为了在流中启动内存传输，我们可以使用函数 **`cudaMemcpyAsync()`** 。

该函数类似于 **`cudaMemcpy()`** 函数，但需要**额外指定用于内存传输的流参数**。

```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, **stream**);
```

<aside>
💡

代码块中的函数调用将通过流 `stream` ，从主机内存中由 `src` 指向的位置复制 `size` 字节数据到设备内存中由 `dst` 指向的位置。

与cudaMemcpy()函数不同的是cudaMemcpy()会阻塞直到内存传输完成。

</aside>

为了让**`cudaMemcpyAsync()`**的**主机内存拷贝**能够**异步执行**，主机缓冲区必须采用**固定（锁定）内存页（pinned and page-locked）。**若使用未固定且未锁定内存页的主机内存，**`cudaMemcpyAsync()`** 会退化为同步执行模式。建议程序使用 **`cudaMallocHost()`** 来分配用于与 GPU 收发数据的缓冲区。

## 流同步 Stream Synchronization

同步流的最简单方法是等待流中无任务。这可以通过两种方式实现： **`cudaStreamSynchronize()`** 函数或 **`cudaStreamQuery()`** 函数。**【注：此处是处理主机host与流的同步】**

下面是使用 **`cudaStreamSynchronize()`** 阻塞式直至流中的所有工作完成：

```cpp
// Wait for the stream to be empty of tasks
cudaStreamSynchronize(stream);

// At this point the stream is done
// and we can access the results of stream operations safely
```

下面是使用 **`cudaStreamQuery()`** 进行非阻塞的快速检查流是否为空：

```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);

switch(status){
		case cudaSuccess:
				// The stream is empty
				std::cout << "The stream is empty" << std::endl;
				break;
		case cudaErrorNotReady:
				// The stream is not empty
				std::cout << "The stream is not empty" << std::endl;
				break;
		default:
				// An error occurred - we should handle this
				break;
}
```

# CUDA事件 CUDA Events

CUDA 事件是一种用于在 CUDA 流中**插入标记**的机制。它们本质上类似于示踪粒子，可用于**跟踪流中任务的进度**。

**场景：**向流中启动两个内核，如果有一个依赖于第一个内核输出的操作，在确认流为空之前，我们将无法安全地启动该操作**（在不使用追踪事件的前提下）**。

**用法：**在第一个内核之后、第二个内核之前向流中**入队一个事件**，可以等待该事件到达**流的前端**。这样就能安全地启动依赖操作。

CUDA 流还保存着**时间信息**，可用于**测算内核启动**和**内存传输**的耗时。

## 创建与销毁CUDA事件

CUDA 事件可通过 **`cudaEventCreate()`** 和 **`cudaEventDestroy()`** 函数进行创建与销毁。

```cpp
cudaEvent_t event;
// Create the event
cudaEventCreate(&event);

// do some work involving the event

// One the work is done and the event is no longer needed
// We can destroy the event
cudaEventDestroy(event);
```

## CUDA流中的计时操作 Timing Operations in CUDA Streams

CUDA 事件可用于为**各类流操作**（包括核函数）的执行计时。当事件**抵达流的前端**时，它会**记录一个时间戳**。

通过将**流中的核函数置于两个事件之间**，我们便能准确测量核函数执行的持续时间：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);

// record the start event
cudaEventRecord(start, stream);

// launch the kernel
kernel<<<grid, block, 0, stream>>>();

// record the stop event
cudaEventRecord(stop, stream);

// wait for the stream to compile
// both events will have been triggered
cudaStreamSynchronize(stream);

// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << "ms" <<std::endl;

// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

## 检查CUDA事件的状态

如同检查流状态一样，可以采用**阻塞或非阻塞**的方式来检查事件的状态。

```cpp
cudaEvent_t event;
cudaStream_t stream;

// Create the stream
cudaStreamCreate(&stream);

// Create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be guaranteed to have completed
// and can launch the dependent task.
cudaEventSynchronize(event);
depentCPUtask();

// wait for the stream to complete
// kernel 2 is guaranteed to have completed
cudaStreammSynchronize(stream);

// destroy the event and stream
cudaEventDestroy(event);
cudaStreamDestroy(stream);
```

CUDA 事件可以通过 **`cudaEventQuery()`** 函数以非阻塞方式检查其完成状态。

```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float* d_data;

// Create some data
cudaMalloc(&d_data, size);
float* h_data = (float*) malloc(size);

// create the streams
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
bool copyStarted = false;

// create the event
cudaEventCreate(&event);

// launch kernel 1 into the stream
kernel<<<grid, block, 0, stream1>>>(d_data, size);

// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel 2 into the stream
kernel<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the GPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2

while(not allCPUWorkDone() || not copyStarted)
{
		doNextChunkOfCPUWork();
		
		// peek to see if kernel 1 has completed
		// if so enqueue a non-blocking copy into stream2
		if(not copyStarted)
		{
				if(cudaEventQuery(event) == cudaSuccess)
				{
						cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
						copyStarted = true;
				}
		}
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

## 来自流的回调函数

CUDA 提供了一种**在主机上从流内启动函数**的机制。目前有两个函数可用： **`cudaLaunchHostFunc()`**  和**`cudaAddCallback()`** 。【**但 `cudaAddCallback()` 计划弃用！**】

函数 **`cudaLaunchHostFunc()`** 的签名如下：

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

- `stream` ：用于启动回调函数的流。
- `func` ：待启动的回调函数。
- `data` ：传递给回调函数的数据指针。

主机函数本身是一个简单的C函数，签名如下：

```cpp
void hostFunction(void *data);
```

其中data参数指向用户**自定义的数据结构 void * 无类型的泛型指针**，函数可对其解释。

**注意：主机函数不得调用任何 CUDA API。**

<aside>
💡

**【注：由于CPU和GPU可同时使用统一内存的同一个指针，GPU正在计算这块内存的时候，CPU（如回调）突然的读写会使程序崩溃。】**

为配合**统一内存**使用，系统提供以下执行保证：

- 在该（主机）函数执行期间，其**所在的流**被视为**空闲（idle）状态**。因此，该函数可以始终安全地使用附加（attached）到其排入的流上的内存。**【防止GPU的数据竞争】**
- 该函数开始执行的效果，等同于对紧接在**该函数之前**、**记录于同一流中的事件**进行**同步**。因此，它会同步那些在该函数之前已经“汇合（joined）”的流。**【“joined”指通过*cudaStreamWaitEvent*建立过依赖关系的跨流等待机制*。*保证Event关联的核函数都执行完毕】**
- 向**任何流中添加设备端任务**，在所有位于其前面的主机函数和流回调**执行完毕之前**，都**不会使该流变为活动（active）状态**。**【即使有任务被添加到了其他流中，只要这些任务通过事件被排序在该函数调用之后，主机函数就仍然可以使用全局附加内存。】**
- 该函数的**执行完成**并不会导致流变为**活动状态**（除非出现上述提及的排队设备任务被触发的情况）。因此，可以通过在流末尾的主机函数中发送信号，来实现流的同步。**【如果函数后或者连续的多个主机函数或流回调之间没有紧跟设备端任务，该流将一直保持空闲状态。】**
</aside>

### 异步错误处理 Asynchronous Error Handling

在 CUDA 流中，错误可能源自**流中的任何操作**，包括**内核启动**和**内存传输**。

这些错误可能直到**流被同步**时才在运行时传回给用户

有两种方法将检查错误的方法：

- **`cudaGetLastError()`** ：函数会返回并清除当前上下文中任何流遇到的最近一次错误。
- **`cudaPeekAtLastError()`** ：函数返回当前上下文中的最后一个错误，但不会清除该错误。

两个函数都返回 `cudaError_t` 类型的错误值。可通过 ***cudaGetErrorName()***和 ***cudaGetErrorString()***函数生成可打印的错误名称。

```cpp
// some work occurs in streams
cudaStreamSynchronize(stream);

// look at the last error but do not clear it 
cudaError_t err = cudaPeekAtLastError();
if(err != cudaSuccess)
{
		printf("Error with name: %s\n", cudaGetErrorName(err));
		printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the lat error and clear it
cudaError_t err2 = cudaGetLastError();
if(err2 != cudaSuccess)
}
		printf("Error with name: %s\n", cudaGetErrorName(err2));
		printf("Error description: %s\n", cudaGetErrorString(err2));
{
if(err2 == err)
{
		printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if(err3 == cudaSuccess)
{
		printf("As expected, cudaGetLastError() cleared the error\n");
}
```

<aside>
💡

当错误发生在同步操作时，尤其是在包含大量操作的流中，通常**难以精确定位错误**在流中的确切位置。一个实用的调试技巧是**设置环境变量** `CUDA_LAUNCH_BLOCKING=1` 后**运行应用程序**。该环境变量的作用是在**每次内核启动后强制同步**，这有助于追踪引发错误的具体内核或传输操作。但需注意，同步操作可能产生**高昂开销，**应用程序的运行速度可能会显著下降。

</aside>

# CUDA流排序 CUDA Stream Order

流中异步操作的**顺序语义**旨在让应用程序开发者能够以安全的方式思考流中操作的执行顺序。

在某些特殊情况下，为优化性能可以**放宽**这些语义约束：

- 在**编程依赖型内核启动场景**中——通过特殊属性与内核启动机制实现两个内核的重叠执行；
- 或在使用 **`cudaMemcpyBatchAsync()`** 函数进行**批量内存传输**时，若运行时能够并发执行非重叠的批量复制操作。

最重要的是，CUDA 流属于所谓的顺序流。这意味着流中操作的执行顺序与其被依序排队的顺序相同，流中的操作**无法超越其他操作提前执行**。

# 阻塞流、非阻塞流与默认流

在 CUDA 中存在两种类型的流：**阻塞流**和**非阻塞流**，阻塞与非阻塞的语义仅指这些**流如何与默认流进行同步。【注意：名称语义可能略显误导，不要混淆】**

默认情况下，使用 **`cudaStreamCreate()`** 创建的流属于阻塞流。

若要创建非阻塞流，必须将 **`cudaStreamCreateWithFlags()`** 函数与 `cudaStreamNonBlocking` 标志结合使用：

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

非阻塞流可以通过常规方式使用 **`cudaStreamDestroy()`** 销毁。

## 传统默认流 Legacy Default Stream

阻塞流与非阻塞流的关键区别在于它们**如何与默认流同步**。

CUDA 提供了一个传统默认流（也称为 **NULL 流**或**流 ID 为 0 的流**），该流在未指定流的内核启动或阻塞式 **`cudaMemcpy()`** 调用时被使用。

这个在所有主机线程间**共享的默认流属于阻塞流**。当操作被提交至此默认流时，它将与所有其他阻塞流同步。**【理解：默认流也属于阻塞流，即当提交操作到默认流时，默认流也将与其他阻塞流同步，等待其他阻塞流执行完毕。】**

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block, 0>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

在上述代码片段中，默认流的行为意味着，即使原则上三个内核能够并发执行，kernel2 仍需等待 kernel1 完成，而 kernel3 亦需等待 kernel2 完成。

以下代码中，使用非阻塞流，原则上所有三个内核均可并发执行。但又因为**无法假设内核执行的任何顺序**，应当执行显式同步以确保内核已完成运算：

```cpp
acudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block, 0>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

## **线程局部的默认流** Pre-thread Default Stream

从 CUDA-7 版本开始，CUDA 允许**每个主机线程拥有独立的默认流**，而非共享传统的默认流。

启用该功能的两种方法：

- 使用 nvcc 编译器选项 `--default-stream per-thread`
- 定义预处理器宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM`

启用此功能后，每个主机线程将具备**独立的默认流**，这些流不与其他流进行同步。

# 显式同步 Explicit Synchronization

有多种方式可以显式地在不同流之间进行同步：

- `cudaDeviceSynchronize()` 会**等待所有主机线程中所有流中所有先前的命令完成**。
- `cudaStreamSynchronize()` 以流作为参数，并**等待指定流中所有先前的命令完成**。它可用于将主机与特定流同步，同时允许其他流在设备上继续执行。
- `cudaStreamWaitEvent()` 以流和事件作为参数，并使得在调用之后添加**到指定流中的所有命令延迟执行，直到给定的事件完成为止**。
- `cudaStreamQuery()` 为应用程序提供了一种方法来知晓**流中所有先前的命令是否已全部完成**。

# 隐式同步 Implicit Synchronization

如果来自**不同流的两个操作之间**提交了任何针对**默认流（NULL 流）**的 CUDA 操作，则这两个操作无法并发执行。**【注：这些流是非阻塞流的情况例外】**

应用程序应遵循**以下准则**以提升内核并发执行的潜力：

- 所有独立操作都应当在**依赖操作之前**发起。
- 应尽可能**延迟任何类型的同步操作**。

# **杂项与高级主题**

## **流优先级 Stream Prioritization**

使用 **`cudaStreamCreateWithPriority()`** 函数创建具备**优先级的流**，该函数接收两个参数：**流句柄**与**优先级数值**(通常设定规则为数值越小代表优先级越高)。

可通过 **`cudaDeviceGetStreamPriorityRange()`** 函数查询特定设备与**上下文环境**中的**有效优先级范围**。流的默认优先级为 0。

```cpp
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the 
// legacy default stream One could also use the option 'cudaStreamNonBlocking' here to 
// create a non-blocking streams

cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);
```

需要指出的是，流的优先级**仅为运行时的参考提示**，通常主要适用于内核启动，可能不适用于内存传输。

流优先级**不会抢占已执行的工作**，也**不能保证任何特定的执行顺序**。

# **基于流捕获的 CUDA 图简介**

**CUDA Graphs with Stream Capture**

通过使用多个流以及结合`cudaStreamWaitEvent`建立的**跨流依赖关系**，应用程序可以构建完整的操作**有向无环图**。某些应用程序可能具有需要在执行过程中多次运行的固定操作序列或 DAG。

CUDA 提供了名为 CUDA 图的功能。CUDA图**创建机制之一**——**流捕获(stream capture)**，通过捕获或创建图，有助于减少主机线程重复调用相同 API 调用链所产生的延迟和 CPU 开销。

用于指定图操作的 API 只需调用一次，生成的图便可多次执行。

CUDA 图按以下方式工作：

1. **应用程序捕获图结构。**这一步骤在首次执行图时完成一次。也可以通过 CUDA 图形 API 手动构建图结构。
2. **图被实例化。**这一步骤在**图捕获完成后执行一次**，用于建立执行图所需的所有运行时结构，以使启动其组件时达到最快速度。
3. 在后续步骤中，**预实例化的图可根据需要重复执行**。由于执行图操作所需的运行时结构均已就位，图执行的 CPU 开销被降至最低。

```cpp
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortkernel(float* out_d, float* in_d)
{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < N) out_d[idx] = 1.23 * in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep < NSTEP; istep++)
{
		if(!graphCreated)
		{
				// Capture the graph
				cudaStreamBeginCapture(stream, cudaStreamCaptureModelGlobal);
				
				// Launch NKERNEL kernels
				for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
				{
						shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
				}
				
				// End the capture
				cudaStreamEndCapture(stream, &graph);
				// Instantiate the graph
				cudaGraphInsatance(&instance, graph, NULL, NULL, 0);
				graphCreated = true;
		}
		
		// Launch the graph
		cudaGraphLaunch(instance, stream);
		
		// Synchronize the stream
		cudaStreamSynchronize(stream);
		
}

```

# **异步执行总结**

本节的核心要点包括：

- 异步 API 使我们能够表达任务的并发执行，为实现各类操作的交叠提供了途径。实际达成的并发程度取决于可用硬件资源与计算能力。
- CUDA 中实现异步执行的关键抽象概念包括流、事件和回调函数。
- 同步可以在事件、流和设备级别进行
- 默认流是一个阻塞流，它会与其他所有阻塞流同步，但不会与非阻塞流同步
- 可以通过 `--default-stream per-thread` 编译器选项或 CUDA_API_PER_THREAD_DEFAULT_STREAM 预处理器宏使用线程局部的默认流来规避默认流行为。
- 可以创建具有不同优先级的流，这些优先级是对运行时的提示，但可能不会在内存传输过程中被遵循。
- CUDA 提供 API 函数以减少或重叠内核启动和内存传输的开销，例如 CUDA 图、批量内存传输和程序化依赖内核启动。